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
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
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
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    output_row_id = tl.program_id(2)
    
    # Shared memory for input tile
    tile_size = kernel_height * kernel_width + 32  # Add padding to avoid bank conflicts
    
    # Calculate output position
    output_h_start = output_row_id * stride_h
    output_w_start = 0
    
    # Process multiple output elements per thread
    for output_w_start in range(0, output_width, OUTPUT_ELEMENTS_PER_BLOCK):
        # Shared memory for input tile
        shared_input = tl.shared_memory(dtype=tl.float32, shape=(tile_size,))
        
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
        
        # Handle bias
        if bias_ptr != 0:
            acc += tl.load(bias_ptr + out_channel_id, mask=out_channel_id < out_channels)
        
        # Loop over input channels and kernel positions
        for g in range(groups):
            group_offset = g * group_size
            
            # Loop over kernel elements
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    ih = output_h_start + kh * dilation_h - padding_h
                    iw = output_w_start + kw * dilation_w - padding_w
                    
                    # Check bounds
                    valid_h = (ih >= 0) & (ih < input_height)
                    valid_w = (iw >= 0) & (iw < input_width)
                    valid = valid_h & valid_w
                    
                    # Load input value
                    if valid:
                        input_idx = batch_id * input_batch_stride + \
                                   (group_offset + 0) * input_channel_stride + \
                                   ih * input_height_stride + \
                                   iw * input_width_stride
                        input_val = tl.load(input_ptr + input_idx)
                    else:
                        input_val = 0.0
                    
                    # Load weight value
                    weight_idx = out_channel_id * weight_out_channel_stride + \
                                (group_offset + 0) * weight_in_channel_stride + \
                                kh * weight_height_stride + \
                                kw * weight_width_stride
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
        
        # Store output
        for i in range(OUTPUT_ELEMENTS_PER_BLOCK):
            if output_w_start + i < output_width:
                output_idx = batch_id * output_batch_stride + \
                           out_channel_id * output_channel_stride + \
                           output_row_id * output_height_stride + \
                           (output_w_start + i) * output_width_stride
                tl.store(output_ptr + output_idx, acc[i])

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
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
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 8
    
    grid = (
        batch_size,
        out_channels,
        (output_height + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Compute strides
    input_batch_stride = input_tensor.stride(0)
    input_channel_stride = input_tensor.stride(1)
    input_height_stride = input_tensor.stride(2)
    input_width_stride = input_tensor.stride(3)
    
    weight_out_channel_stride = weight.stride(0)
    weight_in_channel_stride = weight.stride(1)
    weight_height_stride = weight.stride(2)
    weight_width_stride = weight.stride(3)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_batch_stride,
        input_channel_stride,
        input_height_stride,
        input_width_stride,
        weight_out_channel_stride,
        weight_in_channel_stride,
        weight_height_stride,
        weight_width_stride,
        output_batch_stride,
        output_channel_stride,
        output_height_stride,
        output_width_stride,
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
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using custom Triton kernels for convolution operations.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
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
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)