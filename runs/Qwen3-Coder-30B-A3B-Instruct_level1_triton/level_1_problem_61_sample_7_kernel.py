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
    padding_d,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_element_idx = tl.program_id(2) * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(tl.make_block_ptr(
        input_ptr + batch_idx * in_channels * input_depth * input_height * input_width,
        shape=(in_channels, input_depth, input_height, input_width),
        strides=(input_depth * input_height * input_width, input_height * input_width, input_width, 1),
        block_shape=(CHANNELS_PER_BLOCK, 1, 1, 1)
    ), (CHANNELS_PER_BLOCK, 1, 1, 1))
    
    # Process multiple output elements per thread block
    for output_offset in range(OUTPUT_ELEMENTS_PER_BLOCK):
        if output_element_idx + output_offset >= output_depth * output_height * output_width:
            break
            
        # Calculate output coordinates
        out_z = (output_element_idx + output_offset) // (output_height * output_width)
        out_y = ((output_element_idx + output_offset) % (output_height * output_width)) // output_width
        out_x = (output_element_idx + output_offset) % output_width
        
        # Calculate corresponding input coordinates
        in_z = out_z * stride_d - padding_d
        in_y = out_y * stride_h - padding_h
        in_x = out_x * stride_w - padding_w
        
        # Accumulate output value
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for k_d in range(kernel_depth):
            for k_h in range(kernel_height):
                for k_w in range(kernel_width):
                    # Calculate input coordinates
                    inp_z = in_z + k_d
                    inp_y = in_y + k_h
                    inp_x = in_x + k_w
                    
                    # Check bounds
                    if (inp_z >= 0 and inp_z < input_depth and 
                        inp_y >= 0 and inp_y < input_height and 
                        inp_x >= 0 and inp_x < input_width):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                            batch_idx * in_channels * input_depth * input_height * input_width +
                            channel_idx * input_depth * input_height * input_width +
                            inp_z * input_height * input_width +
                            inp_y * input_width +
                            inp_x)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                            channel_idx * out_channels * kernel_depth * kernel_height * kernel_width +
                            out_channels * kernel_depth * kernel_height * kernel_width +
                            k_d * kernel_height * kernel_width +
                            k_h * kernel_width +
                            k_w)
                        
                        acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + 
            batch_idx * out_channels * output_depth * output_height * output_width +
            channel_idx * output_depth * output_height * output_width +
            out_z * output_height * output_width +
            out_y * output_width +
            out_x,
            acc[0])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_height = (input_height - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_width = (input_width - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        
        # Ensure proper alignment for Triton
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_SIZE = 128
        CHANNELS_PER_BLOCK = 16
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Grid dimensions
        grid_batch = batch_size
        grid_channels = (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
        grid_output = (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
        
        # Launch kernel
        conv_transpose3d_kernel[(grid_batch, grid_channels, grid_output)](
            x,
            weight,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            BLOCK_SIZE,
            CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
            
        return output