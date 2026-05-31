import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_block = tl.program_id(2)
    
    # Calculate output dimensions per group
    channels_per_group = out_channels // groups
    out_channels_start = group_idx * channels_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Process output elements
    output_elements = output_depth * output_height * output_width
    num_output_blocks = (output_elements + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    for output_block in range(num_output_blocks):
        # Calculate global output position
        output_offset = output_block * OUTPUT_ELEMENTS_PER_BLOCK
        if output_offset >= output_elements:
            break
            
        # Calculate output coordinates for this block
        output_pos = output_offset + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
        mask = output_pos < output_elements
        
        # Convert linear output index to 3D coordinates
        out_d = output_pos // (output_height * output_width)
        remaining = output_pos % (output_height * output_width)
        out_h = remaining // output_width
        out_w = remaining % output_width
        
        # Apply stride and padding to get input coordinates
        in_d = out_d * stride_d - pad_d
        in_h = out_h * stride_h - pad_h
        in_w = out_w * stride_w - pad_w
        
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
        
        # Process kernel
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    input_d = in_d + kd
                    input_h = in_h + kh
                    input_w = in_w + kw
                    
                    # Check bounds
                    valid_d = (input_d >= 0) & (input_d < input_depth)
                    valid_h = (input_h >= 0) & (input_h < input_height)
                    valid_w = (input_w >= 0) & (input_w < input_width)
                    valid = valid_d & valid_h & valid_w
                    
                    # Compute weight index
                    weight_idx = (kd * kernel_height * kernel_width + 
                                kh * kernel_width + kw) * channels_per_group
                    
                    # Compute input index
                    input_idx = (batch_idx * in_channels * input_depth * input_height * input_width + 
                               group_idx * channels_per_group * input_depth * input_height * input_width + 
                               input_d * input_height * input_width + 
                               input_h * input_width + 
                               input_w)
                    
                    # Load weights and inputs
                    weight_val = tl.load(weight_ptr + weight_idx + tl.arange(0, channels_per_group), mask=mask)
                    input_val = tl.load(input_ptr + input_idx, mask=valid & mask)
                    
                    # Accumulate
                    acc += input_val * weight_val
        
        # Add bias if enabled
        if bias_enabled:
            bias_vals = tl.load(bias_ptr + out_channels_start + tl.arange(0, channels_per_group), mask=mask)
            acc += bias_vals
        
        # Store output
        output_idx = (batch_idx * out_channels * output_depth * output_height * output_width + 
                     out_channels_start * output_depth * output_height * output_width + 
                     out_d * output_height * output_width + 
                     out_h * output_width + 
                     out_w)
        
        tl.store(output_ptr + output_idx, acc, mask=mask)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    # Get input dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up launch parameters
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 32
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        groups,      # group dimension  
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK  # channel blocks
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Uses custom Triton kernels for performance optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])