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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    channels_per_group,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Grid shape
    grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute output position
    output_h_start = pid_m * BLOCK_SIZE_M
    output_w_start = pid_n * BLOCK_SIZE_N
    
    # Compute group info
    group_id = pid_k // (in_channels // groups)
    group_offset = group_id * channels_per_group
    
    # Loop over K dimension (input channels)
    for k in range(0, grid_k):
        # Load weights
        weight_m = k * BLOCK_SIZE_K
        weight_n = pid_n * BLOCK_SIZE_N
        
        # Check bounds
        if weight_m < in_channels and weight_n < out_channels:
            weight_ptrs = weight_ptr + group_offset * kernel_height * kernel_width * out_channels + \
                         tl.arange(0, BLOCK_SIZE_K)[:, None] * kernel_height * kernel_width * out_channels + \
                         tl.arange(0, BLOCK_SIZE_N)[None, :] * kernel_height * kernel_width
            weights = tl.load(weight_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < in_channels - weight_m) &
                                          (tl.arange(0, BLOCK_SIZE_N)[None, :] < out_channels - weight_n))
            
            # Load input
            input_m = output_h_start * stride_h - padding_h
            input_n = output_w_start * stride_w - padding_w
            
            # Compute convolution
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    ih = input_m + kh * dilation_h
                    iw = input_n + kw * dilation_w
                    
                    # Check bounds for input
                    if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        input_ptr_local = input_ptr + (ih * input_width + iw) * in_channels + weight_m
                        input_vals = tl.load(input_ptr_local, mask=(tl.arange(0, BLOCK_SIZE_K) < in_channels - weight_m))
                        
                        # Multiply and accumulate
                        acc += tl.sum(weights * input_vals[:, None], axis=0)
    
    # Write back results
    output_ptr_local = output_ptr + output_h_start * output_width + output_w_start
    tl.store(output_ptr_local, acc, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < output_height - output_h_start) &
                                       (tl.arange(0, BLOCK_SIZE_N)[None, :] < output_width - output_w_start))

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D convolution.
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = (
        (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )
    
    # Call kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        in_channels // groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Uses custom Triton kernels for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )

# Test code (commented out since we're not supposed to include test code)
"""
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
"""