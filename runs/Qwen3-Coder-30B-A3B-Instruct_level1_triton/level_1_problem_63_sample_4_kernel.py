import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_row_stride,
    input_col_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    output_col_stride,
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
    groups,
    group_size_in,
    group_size_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_row_id = tl.program_id(2)
    
    # Calculate group info
    group_id = out_channel_id // group_size_out
    out_channel_in_group = out_channel_id % group_size_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Grid size for K dimension
    grid_k = (kernel_height * kernel_width * group_size_in + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Loop over K dimension
    for k in range(0, grid_k):
        # Load weights
        weight_offset = (
            group_id * group_size_out * group_size_in * kernel_height * kernel_width +
            out_channel_in_group * group_size_in * kernel_height * kernel_width +
            k * BLOCK_SIZE_K
        )
        weight_ptrs = weight_ptr + weight_offset
        
        # Load input patches
        input_offset = (
            batch_id * input_row_stride * input_height +
            k * BLOCK_SIZE_K
        )
        input_ptrs = input_ptr + input_offset
        
        # Load bias if available
        bias_val = tl.load(bias_ptr + out_channel_id, mask=out_channel_id < out_channels, other=0.0)
        
        # Compute dot product
        for i in range(0, BLOCK_SIZE_M):
            for j in range(0, BLOCK_SIZE_N):
                if i < output_height and j < output_width:
                    # Compute input indices
                    row_start = out_row_id * stride_h + i - padding_h
                    col_start = j * stride_w - padding_w
                    
                    # Accumulate over kernel
                    for ki in range(kernel_height):
                        for kj in range(kernel_width):
                            row_idx = row_start + ki
                            col_idx = col_start + kj
                            
                            # Check bounds
                            if 0 <= row_idx < input_height and 0 <= col_idx < input_width:
                                input_val = tl.load(input_ptr + 
                                    batch_id * input_row_stride * input_height +
                                    group_id * group_size_in * input_height * input_width +
                                    (row_idx * input_width + col_idx) * input_col_stride +
                                    (ki * kernel_width + kj) * BLOCK_SIZE_K,
                                    mask=True, other=0.0)
                                
                                weight_val = tl.load(weight_ptr + 
                                    group_id * group_size_out * group_size_in * kernel_height * kernel_width +
                                    out_channel_in_group * group_size_in * kernel_height * kernel_width +
                                    (ki * kernel_width + kj) * BLOCK_SIZE_K,
                                    mask=True, other=0.0)
                                
                                acc[i, j] += input_val * weight_val
    
    # Write output
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if i < output_height and j < output_width:
                output_offset = (
                    batch_id * output_row_stride * output_height +
                    out_channel_id * output_height * output_width +
                    (out_row_id * output_height + i) * output_width + j
                )
                output_val = acc[i, j] + bias_val
                tl.store(output_ptr + output_offset, output_val)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Handle groups
    group_size_in = in_channels // groups
    group_size_out = out_channels // groups
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(2),
        input_tensor.stride(3),
        weight.stride(2),
        weight.stride(3),
        output.stride(2),
        output.stride(3),
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
        groups,
        group_size_in,
        group_size_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, 
                           stride=(self.stride, self.stride),
                           padding=(self.padding, self.padding),
                           dilation=(self.dilation, self.dilation),
                           groups=self.groups)