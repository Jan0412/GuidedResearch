import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, in_channels, out_channels, height, width, 
    kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    has_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block ID for this program
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Grid shape
    grid_m = tl.cdiv(height, BLOCK_SIZE_M)
    grid_n = tl.cdiv(width, BLOCK_SIZE_N)
    
    # Tile indices
    tile_m = pid_m * BLOCK_SIZE_M
    tile_n = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels)
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input tile
        input_tile = tl.load(
            input_ptr +
            tile_m * input_stride_2 +
            k * input_stride_1 +
            tile_n * input_stride_3,
            mask=(tile_m < height) & (tile_n < width),
            other=0.0
        )
        
        # Load weight tile
        weight_tile = tl.load(
            weight_ptr +
            k * weight_stride_1 +
            (tile_m % kernel_h) * weight_stride_2 +
            (tile_n % kernel_w) * weight_stride_3,
            mask=(k < in_channels) & (tile_m < kernel_h) & (tile_n < kernel_w),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Add bias if needed
    if has_bias:
        bias = tl.load(bias_ptr + pid_m * output_stride_1, mask=(pid_m < out_channels))
        acc += bias
    
    # Store result
    tl.store(
        output_ptr +
        tile_m * output_stride_2 +
        pid_n * output_stride_3,
        acc,
        mask=(tile_m < height) & (pid_n < width)
    )

def triton_conv2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    out_width = (width + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Define grid
    grid = (
        triton.cdiv(out_height, BLOCK_SIZE_M),
        triton.cdiv(out_width, BLOCK_SIZE_N)
    )
    
    # Call kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        in_channels,
        out_channels,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        padding,
        padding,
        stride,
        stride,
        dilation,
        dilation,
        bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton implementation
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )