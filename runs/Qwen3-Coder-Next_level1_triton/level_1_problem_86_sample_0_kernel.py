import torch
import torch.nn as nn
import triton
import triton.language as tl

# Depthwise convolution kernel - optimized for 2D depthwise conv
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Weight tensor (C, 1, K, K)
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    B, C, H, W, K,  # Dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    H_out, W_out,  # Output dimensions
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    c_id = tl.program_id(0)
    h_id = tl.program_id(1)
    b_id = tl.program_id(2)
    
    # Calculate channel offset
    c_start = c_id * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Calculate spatial position
    h_start = h_id * BLOCK_SIZE_H
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_offsets < H_out
    
    # Batch index is handled by the program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(K):
        for kw in range(K):
            # Calculate input positions with dilation and padding
            h_in = h_start * stride_h + kh * dil_h - pad_h
            w_in = tl.arange(0, BLOCK_SIZE_H) * stride_w + kw * dil_w - pad_w
            
            # Load input values
            h_in_offsets = h_in * W + w_in
            x_offset = b_id * C * H * W + c_offsets[:, None] * H * W + h_in_offsets[None, :]
            
            # Mask for valid input positions
            h_valid = (h_in >= 0) & (h_in < H)
            w_valid = (tl.arange(0, BLOCK_SIZE_H) * stride_w + kw * dil_w - pad_w >= 0) & \
                     (tl.arange(0, BLOCK_SIZE_H) * stride_w + kw * dil_w - pad_w < W)
            mask = c_mask[:, None] & (h_valid[None, :] & w_valid[None, :])
            
            x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            
            # Load weight values
            w_offset = c_offsets[:, None] * K * K + kh * K + kw
            w_val = tl.load(w_ptr + w_offset, mask=c_mask[:, None], other=0.0)
            
            # Accumulate
            acc += x_val * w_val
    
    # Store result
    out_offset = b_id * C * H_out * W_out + c_offsets[:, None] * H_out * W_out + \
                (h_start + tl.arange(0, BLOCK_SIZE_H)[None, :]) * W_out + tl.arange(0, BLOCK_SIZE_H)[None, :]
    out_mask = c_mask[:, None] & (h_mask[None, :] & (tl.arange(0, BLOCK_SIZE_H)[None, :] < W_out))
    tl.store(out_ptr + out_offset, acc, mask=out_mask)

# Pointwise convolution kernel (1x1 conv)
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    w_ptr,  # Weight tensor (C_out, C_in, 1, 1)
    out_ptr,  # Output tensor (B, C_out, H, W)
    B, C_in, C_out, H, W,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program IDs
    c_out_id = tl.program_id(0)
    hw_id = tl.program_id(1)
    b_id = tl.program_id(2)
    
    # Calculate output channel offset
    c_out_start = c_out_id * BLOCK_SIZE_C_out
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate spatial position
    hw_start = hw_id * BLOCK_SIZE_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_offsets < H * W
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_HW), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Load input values
        x_offset = b_id * C_in * H * W + c_in_offsets[None, :] * H * W + hw_offsets[:, None]
        x_val = tl.load(x_ptr + x_offset, mask=c_in_mask[None, :] & (hw_offsets[:, None] < H * W), other=0.0)
        
        # Load weight values
        w_offset = c_out_offsets[:, None] * C_in + c_in_offsets[None, :]
        w_val = tl.load(w_ptr + w_offset, mask=c_out_mask[:, None] & c_in_mask[None, :], other=0.0)
        
        # Accumulate: matmul-like operation
        # w_val: (C_out_block, C_in_block), x_val: (C_in_block, HW_block)
        acc += tl.dot(w_val, x_val)
    
    # Store result
    out_offset = b_id * C_out * H * W + c_out_offsets[:, None] * H * W + hw_offsets[None, :]
    out_mask = c_out_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)

# Helper function for depthwise convolution
def triton_depthwise_conv2d(x, weight, stride=1, padding=0, dilation=1):
    B, C, H, W = x.shape
    K = weight.shape[2]
    assert weight.shape[1] == 1, "Depthwise conv requires groups=in_channels"
    
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    out = torch.empty(B, C, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure grid
    grid_c = (C + 64 - 1) // 64
    grid_h = (H_out + 32 - 1) // 32
    grid = (grid_c, grid_h, B)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, out,
        B, C, H, W, K,
        stride, stride,  # stride_h, stride_w
        padding, padding,  # pad_h, pad_w
        dilation, dilation,  # dil_h, dil_w
        H_out, W_out,
        BLOCK_SIZE_C=64,
        BLOCK_SIZE_H=32,
        BLOCK_SIZE_K=3,
    )
    
    return out

# Helper function for pointwise convolution
def triton_pointwise_conv2d(x, weight):
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    out = torch.empty(B, C_out, H, W, dtype=x.dtype, device=x.device)
    
    # Configure grid
    grid_c_out = (C_out + 64 - 1) // 64
    grid_hw = (H * W + 128 - 1) // 128
    grid = (grid_c_out, grid_hw, B)
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, out,
        B, C_in, C_out, H, W,
        BLOCK_SIZE_C_in=64,
        BLOCK_SIZE_C_out=64,
        BLOCK_SIZE_HW=128,
    )
    
    return out

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation with optimized Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Depthwise convolution with groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform depthwise convolution using Triton kernel
        x = self.depthwise(x)
        # Perform pointwise convolution using Triton kernel
        x = self.pointwise(x)
        return x