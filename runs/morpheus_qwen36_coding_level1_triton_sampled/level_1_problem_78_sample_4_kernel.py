import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    in_ptr, out_ptr, weight_ptr, bias_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w, stride_h, stride_w, pad_h, pad_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Each program handles a tile of output spatial dimensions for a block of output channels
    pid = tl.program_id(0)
    num_tiles_h = (H_out + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (W_out + BLOCK_W - 1) // BLOCK_W
    num_tiles = num_tiles_h * num_tiles_w
    
    # Map pid to channel block and spatial tile
    c_out_start = (pid // num_tiles) % C_out
    c_out_end = min(c_out_start + BLOCK_C, C_out)
    c_out_idx = c_out_start + tl.arange(0, BLOCK_C)
    
    tile_h = (pid // (num_tiles_w * C_out)) // num_tiles_h
    tile_w = (pid // (num_tiles_w * C_out)) % num_tiles_w
    
    h_start = tile_h * BLOCK_H
    w_start = tile_w * BLOCK_W
    
    h_idx = h_start + tl.arange(0, BLOCK_H)
    w_idx = w_start + tl.arange(0, BLOCK_W)
    
    h_mask = h_idx < H_out
    w_mask = w_idx < W_out
    c_out_mask = c_out_idx < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels in chunks
    for c_in_start in range(0, C_in, BLOCK_C):
        c_in_end = min(c_in_start + BLOCK_C, C_in)
        c_in_range = c_in_end - c_in_start
        
        # Load input tile
        # Input coordinates: h_in = h_out // stride_h - kh + pad_h
        #                   w_in = w_out // stride_w - kw + pad_w
        # We compute base indices for the output tile
        h_base = h_start // stride_h
        w_base = w_start // stride_w
        
        # Load input tensor tile: shape (BLOCK_C, BLOCK_H, BLOCK_W)
        # We need to gather from input based on kh, kw
        # To optimize, we load the entire relevant input region for this spatial tile
        # and kernel, then compute. But for simplicity and correctness, we'll compute directly.
        
        # For each kh, kw, compute corresponding h_in, w_in
        for kh in range(K_h):
            for kw in range(K_w):
                h_in = h_base - kh + pad_h
                w_in = w_base - kw + pad_w
                
                # Compute input indices
                h_in_idx = h_in + tl.arange(0, BLOCK_H) * stride_h
                w_in_idx = w_in + tl.arange(0, BLOCK_W) * stride_w
                
                # Mask for valid input coordinates
                h_in_mask = (h_in_idx >= 0) & (h_in_idx < H_in)
                w_in_mask = (w_in_idx >= 0) & (w_in_idx < W_in)
                valid_mask = h_in_mask & w_in_mask
                
                # Load input values
                in_ptr_offset = h_in_idx[:, None, None] * W_in * C_in + w_in_idx[None, :, None] * C_in + tl.arange(0, BLOCK_C)[None, None, :]
                in_vals = tl.load(in_ptr + in_ptr_offset, mask=valid_mask[:, :, None], other=0.0)
                
                # Load kernel values: shape (BLOCK_C, BLOCK_C, 1, 1) for this kh, kw
                weight_ptr_offset = c_out_idx[:, None, None, None] * C_in * K_h * K_w + (c_in_start + tl.arange(0, BLOCK_C))[None, :, None, None] * K_h * K_w + kh * K_w + kw
                weight_vals = tl.load(weight_ptr + weight_ptr_offset, mask=c_out_mask[:, None, None, None] & tl.arange(0, BLOCK_C)[None, :, None, None] < C_in, other=0.0)
                
                # Accumulate
                acc += in_vals * weight_vals
    
    # Apply output masks
    acc = acc * c_out_mask[:, None, None] * h_mask[None, :, None] * w_mask[None, None, :]
    
    # Store output
    out_ptr_offset = c_out_idx[:, None, None] * H_out * W_out + h_idx[None, :, None] * W_out + w_idx[None, None, :]
    tl.store(out_ptr + out_ptr_offset, acc, mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])
    
    # Add bias if present
    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr + c_out_idx, mask=c_out_mask, other=0.0)
        out_ptr_offset_bias = c_out_idx[:, None, None] * H_out * W_out + h_idx[None, :, None] * W_out + w_idx[None, None, :]
        acc += bias_vals[:, None, None]
        tl.store(out_ptr + out_ptr_offset_bias, acc, mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0)
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w
    
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C = 8
    
    num_tiles_h = (H_out + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (W_out + BLOCK_W - 1) // BLOCK_W
    num_channel_blocks = (C_out + BLOCK_C - 1) // BLOCK_C
    grid = (num_channel_blocks * num_tiles_h * num_tiles_w,)
    
    conv_transpose2d_kernel[grid](
        x, out, weight, bias,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w, stride_h, stride_w, pad_h, pad_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        return triton_conv_transpose2d(x, weight, bias, self.stride, self.padding)