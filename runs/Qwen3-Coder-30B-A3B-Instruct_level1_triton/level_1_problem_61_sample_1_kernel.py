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
    
    # Calculate how many output elements this block handles
    output_elements_start = tl.program_id(2) * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Process output elements in chunks
    for output_element_offset in range(OUTPUT_ELEMENTS_PER_BLOCK):
        output_element_idx = output_elements_start + output_element_offset
        
        if output_element_idx >= output_depth * output_height * output_width:
            break
            
        # Calculate output coordinates
        out_z = output_element_idx // (output_height * output_width)
        out_y = (output_element_idx % (output_height * output_width)) // output_width
        out_x = output_element_idx % output_width
        
        # Initialize accumulator
        acc = 0.0
        
        # Loop over kernel dimensions and input channels
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    for ic in range(in_channels):
                        # Calculate input coordinates
                        input_z = out_z * stride_d - padding_d + kd
                        input_y = out_y * stride_h - padding_h + kh
                        input_x = out_x * stride_w - padding_w + kw
                        
                        # Check bounds
                        if (input_z >= 0 and input_z < input_depth and 
                            input_y >= 0 and input_y < input_height and 
                            input_x >= 0 and input_x < input_width):
                            
                            # Calculate input and weight indices
                            input_idx = (batch_idx * in_channels * input_depth * input_height * input_width + 
                                       ic * input_depth * input_height * input_width + 
                                       input_z * input_height * input_width + 
                                       input_y * input_width + 
                                       input_x)
                            
                            weight_idx = (channel_idx * in_channels * kernel_depth * kernel_height * kernel_width + 
                                        ic * kernel_depth * kernel_height * kernel_width + 
                                        kd * kernel_height * kernel_width + 
                                        kh * kernel_width + 
                                        kw)
                            
                            input_val = tl.load(input_ptr + input_idx)
                            weight_val = tl.load(weight_ptr + weight_idx)
                            acc += input_val * weight_val
        
        # Store result
        output_idx = (batch_idx * out_channels * output_depth * output_height * output_width + 
                     channel_idx * output_depth * output_height * output_width + 
                     out_z * output_height * output_width + 
                     out_y * output_width + 
                     out_x)
        
        tl.store(output_ptr + output_idx, acc)

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
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        # Create output tensor
        output = torch.zeros(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define parameters for kernel launch
        BLOCK_SIZE = 16
        CHANNELS_PER_BLOCK = 1
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
            (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x,
            self.weight,
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
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output