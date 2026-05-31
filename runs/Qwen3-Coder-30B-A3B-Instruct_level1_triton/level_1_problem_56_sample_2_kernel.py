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
    group_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Calculate output width index
    out_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_offset = g * group_size
        weight_offset = out_ch_idx * group_size * kernel_height * kernel_width + group_offset * kernel_height * kernel_width
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position with stride and padding
                ih = out_h_idx * stride_h - padding_h + kh * dilation_h
                iw = out_w_idx * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Calculate input offset
                    input_offset = batch_idx * in_channels * input_height * input_width + \
                                  group_offset * input_height * input_width + \
                                  ih * input_width + iw
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + weight_offset + kh * kernel_width + kw)
                    
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Accumulate
                    acc += weight_val * input_val
                
                # Move to next weight
                weight_offset += 1
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Write output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_ch_idx * output_height * output_width + \
                   out_h_idx * output_width + out_w_idx
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Ensure tensors are contiguous and on CUDA
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    if bias is not None:
        bias = bias.contiguous().cuda()
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, dtype=torch.float32, device='cuda')
    
    # Define block size and grid
    BLOCK_SIZE = 32
    GROUP_SIZE = 8
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Uses custom Triton kernel for performance optimization.
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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
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

# Note: The current implementation has limitations compared to PyTorch's native conv2d.
# A more robust implementation would require additional optimizations like:
# 1. Tiling for better memory access patterns
# 2. Shared memory usage for input tiles
# 3. Better handling of boundary conditions
# 4. Support for different data types
# For production use, this would benefit from extensive benchmarking and tuning.