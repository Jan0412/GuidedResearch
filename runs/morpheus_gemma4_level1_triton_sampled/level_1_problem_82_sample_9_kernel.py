import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W, K, S, P, OH, OW,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wk, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Parallelize over Batch * Channel and Output Height/Width blocks
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    # Output indices for this block
    oh_range = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_range = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Masks for output boundaries
    oh_mask = oh_range < OH
    ow_mask = ow_range < OW

    # Load bias for the current channel
    # b_ptr might be null if bias=False, but we handle it by passing a dummy or checking
    bias = tl.load(b_ptr + c) if b_ptr is not None else 0.0
    
    # Initialize accumulator with bias
    acc = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), bias, dtype=tl.float32)

    # Loop over the kernel window
    for kh in range(K):
        for kw in range(K):
            # Calculate input coordinates
            h_idx = oh_range * S + kh - P
            w_idx = ow_range * S + kw - P
            
            # Boundary check for padding
            h_mask = (h_idx >= 0) & (h_idx < H)
            w_mask = (w_idx >= 0) & (w_idx < W)
            
            # Load input value (masked for padding)
            # x_ptr + b*stride_xb + c*stride_xc + h_idx*stride_xh + w_idx*stride_xw
            # We need to broadcast indices to (BLOCK_SIZE_H, BLOCK_SIZE_W)
            x_offsets = (
                b * stride_xb + 
                c * stride_xc + 
                h_idx[:, None] * stride_xh + 
                w_idx[None, :] * stride_xw
            )
            
            # Mask for the 2D block
            block_mask = h_mask[:, None] & w_mask[None, :]
            x_val = tl.load(x_ptr + x_offsets, mask=block_mask, other=0.0)
            
            # Load weight value
            # w_ptr + c*stride_wc + kh*stride_wk + kw*stride_ww
            w_val = tl.load(w_ptr + c * stride_wc + kh * stride_wk + kw * stride_ww)
            
            acc += x_val * w_val

    # Store the result
    out_offsets = (
        b * stride_ob + 
        c * stride_oc + 
        oh_range[:, None] * stride_oh + 
        ow_range[None, :] * stride_ow
    )
    
    tl.store(out_ptr + out_offsets, acc, mask=oh_mask[:, None] & ow_mask[None, :])


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # x: (B, C, H, W)
    # weight: (C, 1, K, K)
    # bias: (C,) or None
    B, C, H, W = x.shape
    K = weight.shape[2]
    
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1
    
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=torch.float32)
    
    if bias is not None:
        bias = bias.contiguous().float()
    else:
        bias = torch.zeros(C, device=x.device, dtype=torch.float32)

    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wc, _, stride_wk, stride_ww = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()
    
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    grid = (B * C, (OH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C, H, W, K, stride, padding, OH, OW,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wk, stride_ww,
        stride_ob, stride_oc, stride_oh, stride_ow,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        # We keep the Conv2d layer to manage weights and bias parameters
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton implementation instead of the PyTorch operator
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding
        )