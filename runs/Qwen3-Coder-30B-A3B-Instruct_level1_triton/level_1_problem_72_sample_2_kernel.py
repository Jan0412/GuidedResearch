import torch
import torch.nn as nn
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
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_block = tl.program_id(2)
    
    # Calculate output positions
    output_elements_per_thread = OUTPUT_ELEMENTS_PER_BLOCK // BLOCK_SIZE
    output_element_offset = tl.program_id(3) * OUTPUT_ELEMENTS_PER_BLOCK + tl.arange(0, BLOCK_SIZE) % output_elements_per_thread
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr + batch_idx * in_channels * input_depth * input_height * input_width, 
                                BLOCK_SIZE, 1)
    
    # Process multiple output elements per thread
    for output_idx in range(output_elements_per_thread):
        # Calculate global output position
        global_output_idx = tl.program_id(3) * OUTPUT_ELEMENTS_PER_BLOCK + output_idx
        
        if global_output_idx >= output_depth * output_height * output_width:
            break
            
        # Convert linear index to 3D coordinates
        out_z = global_output_idx // (output_height * output_width)
        out_y = (global_output_idx % (output_height * output_width)) // output_width
        out_x = global_output_idx % output_width
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # For each input channel in this group
        for c in range(channel_block * CHANNELS_PER_BLOCK, 
                      min((channel_block + 1) * CHANNELS_PER_BLOCK, in_channels)):
            # Calculate kernel positions
            for kd in range(kernel_depth):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        # Calculate input position
                        input_z = out_z * stride_depth - padding_depth + kd
                        input_y = out_y * stride_height - padding_height + kh
                        input_x = out_x * stride_width - padding_width + kw
                        
                        # Check bounds
                        if (input_z >= 0 and input_z < input_depth and
                            input_y >= 0 and input_y < input_height and
                            input_x >= 0 and input_x < input_width):
                            
                            # Load input value
                            input_val = tl.load(input_ptr + 
                                              batch_idx * in_channels * input_depth * input_height * input_width +
                                              c * input_depth * input_height * input_width +
                                              input_z * input_height * input_width +
                                              input_y * input_width +
                                              input_x)
                            
                            # Load weight value
                            weight_val = tl.load(weight_ptr + 
                                               group_idx * group_size * kernel_depth * kernel_height * kernel_width * out_channels +
                                               (c % group_size) * kernel_depth * kernel_height * kernel_width * out_channels +
                                               kd * kernel_height * kernel_width * out_channels +
                                               kh * kernel_width * out_channels +
                                               kw * out_channels +
                                               (c // group_size) * out_channels)
                            
                            acc += input_val * weight_val
        
        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + (c // group_size) * out_channels)
            acc += bias_val
            
        # Store result
        tl.store(output_ptr + 
                batch_idx * out_channels * output_depth * output_height * output_width +
                (c // group_size) * output_depth * output_height * output_width +
                out_z * output_height * output_width +
                out_y * output_width +
                out_x, acc)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 1024
    
    # Calculate grid dimensions
    grid_batch = batch_size
    grid_groups = groups
    grid_channel_blocks = (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    grid_output_elements = (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    grid = (grid_batch, grid_groups, grid_channel_blocks, grid_output_elements)
    
    # Launch kernel
    group_size = in_channels // groups
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    conv_transpose3d_kernel[grid](
        input_tensor.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
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
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups,
        group_size,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.
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

# For compatibility with existing interface
def get_inputs():
    batch_size = 8
    in_channels = 32
    out_channels = 32
    kernel_size = (3, 5, 7)
    depth = 12
    height = 24
    width = 48
    stride = (2, 2, 2)
    padding = (1, 2, 3)
    output_padding = (1, 1, 1)
    groups = 4
    
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [32, 32, (3, 5, 7), (2, 2, 2), (1, 2, 3), (1, 1, 1), 4]