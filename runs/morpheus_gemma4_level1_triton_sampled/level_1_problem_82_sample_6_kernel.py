import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, 
    w_ptr, 
    b_ptr, 
    out_ptr, 
    B, C, H, W, 
    KH, KW, 
    S, P, 
    H_out, W_out, 
    stride_h, stride_w, 
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
):
    # pid_0 represents the (batch, channel) combination
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)

    batch_idx = pid_0 // C
    channel_idx = pid_0 % C

    # Output coordinates for this block
    oh_start = pid_1 * BLOCK_H
    ow_start = pid_2 * BLOCK_W
    
    oh = oh_start + tl.arange(0, BLOCK_H)
    ow = ow_start + tl.arange(0, BLOCK_W)

    # Mask for output boundaries
    oh_mask = oh < H_out
    ow_mask = ow < W_out
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    # Pointers to the start of the current batch and channel
    # x: (B, C, H, W)
    x_base_ptr = x_ptr + batch_idx * (C * H * W) + channel_idx * (H * W)
    # w: (C, 1, KH, KW)
    w_base_ptr = w_ptr + channel_idx * (KH * KW)
    # out: (B, C, H_out, W_out)
    out_base_ptr = out_ptr + batch_idx * (C * H_out * W_out) + channel_idx * (H_out * W_out)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over the kernel window
    for kh in range(0, KH):
        for kw in range(0, KW):
            # Calculate input coordinates
            # x_idx = oh * S + kh - P
            h_idx = oh * S + kh - P
            w_idx = ow * S + kw - P
            
            # Mask for input boundaries (padding)
            h_mask = (h_idx >= 0) & (h_idx < H)
            w_mask = (w_idx >= 0) & (w_idx < W)
            in_mask = h_mask[:, None] & w_mask[None, :]
            
            # Load input patch
            # Offset: h_idx * W + w_idx
            # We use [:, None] and [None, :] to broadcast to (BLOCK_H, BLOCK_W)
            input_offsets = h_idx[:, None] * W + w_idx[None, :]
            input_val = tl.load(x_base_ptr + input_offsets, mask=in_mask, other=0.0)
            
            # Load weight for this specific kernel position
            weight_val = tl.load(w_base_ptr + kh * KW + kw)
            
            acc += input_val * weight_val

    # Add bias
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + channel_idx)
        acc += bias_val

    # Store result
    out_offsets = oh[:, None] * W_out + ow[None, :]
    tl.store(out_base_ptr + out_offsets, acc, mask=out_mask)


def triton_depthwise_conv(x, weight, bias, stride=1, padding=0):
    B, C, H, W = x.shape
    KH, KW = weight.shape[2:]
    S = stride
    P = padding

    H_out = (H + 2 * P - KH) // S + 1
    W_out = (W + 2 * P - KW) // S + 1

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    BLOCK_H = 16
    BLOCK_W = 16

    grid = (B * C, triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))

    depthwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C, H, W,
        KH, KW,
        S, P,
        H_out, W_out,
        W, W, # stride_h/w not explicitly used as offsets but passed for consistency
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # We keep the nn.Conv2d to manage parameters (weights and bias)
        self.conv2d = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Use the custom Triton implementation
        return triton_depthwise_conv(
            x, 
            weight, 
            bias, 
            stride=self.stride, 
            padding=self.padding
        )