import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,          # [B, C_in, H, W]
    w_ptr,          # [C_out, C_in, K_h, K_w]
    out_ptr,        # [B, C_out, H_out, W_out]
    # Shapes
    B, C_in, H, W,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K_h: tl.constexpr,
    BLOCK_K_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index
    
    # Calculate output coordinates
    out_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) % BLOCK_SIZE
    out_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) // BLOCK_SIZE
    
    # Check bounds
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h & mask_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute input spatial coordinates
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w
    
    # Loop over output channels in blocks
    c_out_start = pid_c_out * BLOCK_C_out
    
    # Loop over input channels
    for c_in in range(C_in):
        # Loop over kernel height
        for kh in range(K_h):
            # Loop over kernel width
            for kw in range(K_w):
                # Compute input coordinates
                h_idx = in_h + kh
                w_idx = in_w + kw
                
                # Check if within input bounds
                valid_h = (h_idx >= 0) & (h_idx < H)
                valid_w = (w_idx >= 0) & (w_idx < W)
                valid = valid_h & valid_w
                
                # Compute input pointer offset
                # x_ptr shape: [B, C_in, H, W]
                x_offset = (pid_b * C_in * H * W + 
                           c_in * H * W + 
                           h_idx * W + 
                           w_idx)
                
                # Load input value
                x_val = tl.load(x_ptr + x_offset, 
                              mask=valid, 
                              other=0.0)
                
                # Compute weight pointer offset
                # w_ptr shape: [C_out, C_in, K_h, K_w]
                w_offset = (c_out_start * C_in * K_h * K_w + 
                           c_in * K_h * K_w + 
                           kh * K_w + 
                           kw)
                
                # Load weight value
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += tl.where(valid, x_val * w_val, 0.0)
    
    # Store result
    out_offset = (pid_b * C_out * H_out * W_out + 
                 pid_c_out * H_out * W_out + 
                 out_h * W_out + 
                 out_w)
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv2d(x, weight, stride=4, padding=2):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE = 32
    BLOCK_C_out = min(16, C_out)
    BLOCK_C_in = C_in
    BLOCK_K_h = K_h
    BLOCK_K_w = K_w
    
    # Define grid dimensions
    grid = (B, 
            (C_out + BLOCK_C_out - 1) // BLOCK_C_out,
            (H_out + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride, stride,
        padding, padding,
        H_out, W_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K_h=BLOCK_K_h,
        BLOCK_K_w=BLOCK_K_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Use custom Triton convolution kernel
        # Note: For simplicity, we're using the same convolution logic but could be optimized further
        return triton_conv2d(x, self.conv1.weight, stride=4, padding=2)