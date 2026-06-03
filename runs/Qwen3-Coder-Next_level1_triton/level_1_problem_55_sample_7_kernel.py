import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    N, H, W,  # Input dimensions: batch, height, width
    C_in, C_out,  # Input and output channels
    K,  # Kernel size
    stride, padding, dilation,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction
    BLOCK_H: tl.constexpr,  # Block size for height
    BLOCK_W: tl.constexpr,  # Block size for width
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    pid_h = tl.program_id(2)  # For output height
    pid_w = tl.program_id(3)  # For output width
    
    # Calculate output position
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Calculate input position for the top-left corner of the first output element
    in_h_start = out_h * stride - pad_h
    in_w_start = out_w * stride - pad_w
    
    # Create accumulators for output
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_K):
        # Create range for input channels
        c_in_offsets = c_in + tl.arange(0, BLOCK_SIZE_K)
        c_in_mask = c_in_offsets < C_in
        
        # Loop over kernel height
        for kh in range(K):
            # Calculate input height position
            in_h = in_h_start + kh * dil_h
            h_valid = (in_h >= 0) & (in_h < H)
            
            # Loop over kernel width
            for kw in range(K):
                # Calculate input width position
                in_w = in_w_start + kw * dil_w
                w_valid = (in_w >= 0) & (in_w < W)
                
                # Load input data
                in_offsets = (
                    pid_n * (C_in * H * W) + 
                    c_in_offsets[:, None, None] * (H * W) + 
                    in_h * W + 
                    in_w
                )
                
                # Mask for valid input elements
                input_mask = (
                    c_in_mask[:, None, None] & 
                    h_valid[None, :, None] & 
                    w_valid[None, None, :]
                )
                
                # Load input values
                x = tl.load(
                    input_ptr + in_offsets,
                    mask=input_mask,
                    other=0.0
                )
                
                # Load weight data
                w_offsets = (
                    pid_m * BLOCK_SIZE_M * (C_in * K * K) +
                    c_in_offsets[:, None, None] * (K * K) +
                    kh * K +
                    kw
                )
                
                w_mask = (
                    tl.arange(0, BLOCK_SIZE_M)[:, None, None] < C_out &
                    c_in_mask[:, None, None] &
                    (kh < K) &
                    (kw < K)
                )
                
                w = tl.load(
                    weight_ptr + w_offsets,
                    mask=w_mask,
                    other=0.0
                )
                
                # Accumulate convolution
                output += tl.sum(x * w, axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bias_mask = bias_offsets < C_out
        bias_val = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        output += bias_val[:, None, None]
    
    # Store output
    out_offsets = (
        pid_n * (C_out * H_out * W_out) +
        (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None, None]) * (H_out * W_out) +
        (out_h + tl.arange(0, BLOCK_H)[None, :, None]) * W_out +
        (out_w + tl.arange(0, BLOCK_W)[None, None, :])
    )
    
    out_mask = (
        (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None, None]) < C_out &
        (out_h + tl.arange(0, BLOCK_H)[None, :, None]) < H_out &
        (out_w + tl.arange(0, BLOCK_W)[None, None, :]) < W_out
    )
    
    tl.store(output_ptr + out_offsets, output, mask=out_mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections
        
    Returns:
        Output tensor of shape (batch_size, out_channels, height_out, width_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Ensure kernel is square
    assert K_h == K_w, "Only square kernels are supported in this implementation."
    K = K_h
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set up block sizes for optimization
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 1   # Batch per block (1 for simplicity)
    BLOCK_SIZE_K = 8   # Input channels per block
    
    # Block sizes for spatial dimensions
    BLOCK_H = 4
    BLOCK_W = 8
    
    # Grid dimensions
    grid = (
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # m blocks
        N,  # n blocks
        (H_out + BLOCK_H - 1) // BLOCK_H,  # h blocks
        (W_out + BLOCK_W - 1) // BLOCK_W   # w blocks
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, out, weight, bias,
        N, H, W,
        C_in, C_out,
        K,
        stride, padding, dilation,
        stride, stride,
        padding, padding,
        dilation, dilation,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model using Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same convolution layer
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Extract parameters from the original conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Call the Triton convolution implementation
        return triton_conv2d(
            x, weight, bias,
            stride=self.conv2d.stride[0],
            padding=self.conv2d.padding[0],
            dilation=self.conv2d.dilation[0],
            groups=self.conv2d.groups
        )