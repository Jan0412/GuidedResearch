import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
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
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_depth_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Each thread handles one output element
    output_element_idx = tl.program_id(3) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_element_idx < output_elements
    
    # Calculate spatial coordinates for this output element
    out_w = output_element_idx % output_width
    out_h = (output_element_idx // output_width) % output_height
    out_d = (output_element_idx // (output_width * output_height)) % output_depth
    
    # Apply stride and padding to get input coordinates
    in_d_start = out_d * stride_d - padding_d
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for g in range(groups):
        # Calculate channel offset for this group
        channel_offset = g * group_size
        
        # Loop over kernel dimensions
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    in_d = in_d_start + kd * dilation_d
                    in_h = in_h_start + kh * dilation_h
                    in_w = in_w_start + kw * dilation_w
                    
                    # Check if input coordinate is valid
                    if (in_d >= 0 and in_d < input_depth and 
                        in_h >= 0 and in_h < input_height and 
                        in_w >= 0 and in_w < input_width):
                        
                        # Calculate input index
                        input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                   (channel_offset + (g * group_size)) * (input_depth * input_height * input_width) +
                                   in_d * (input_height * input_width) + 
                                   in_h * input_width + 
                                   in_w)
                        
                        # Calculate weight index
                        weight_idx = (out_channel_idx * (groups * kernel_depth * kernel_height * kernel_width * group_size) +
                                    g * (kernel_depth * kernel_height * kernel_width * group_size) +
                                    kd * (kernel_height * kernel_width * group_size) +
                                    kh * (kernel_width * group_size) +
                                    kw * group_size +
                                    (g * group_size))
                        
                        # Load input and weight values
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_channel_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Store output
    output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                 out_channel_idx * (output_depth * output_height * output_width) +
                 out_d * (output_height * output_width) +
                 out_h * output_width +
                 out_w)
    
    tl.store(output_ptr + output_idx, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.shape[0]
        input_depth = x.shape[2]
        input_height = x.shape[3]
        input_width = x.shape[4]
        
        # Calculate output dimensions
        kernel_depth, kernel_height, kernel_width = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        
        output_depth = (input_depth + 2 * padding_d - (dilation_d * (kernel_depth - 1) + 1)) // stride_d + 1
        output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
        output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Prepare pointers
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        output_ptr = output.data_ptr()
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Define constants
        BLOCK_SIZE = 128
        GROUP_SIZE = 8
        
        # Calculate grid dimensions
        grid = (
            batch_size,  # batch dimension
            self.out_channels,  # output channel dimension
            output_depth,  # output depth dimension
            (output_depth * output_height * output_width + BLOCK_SIZE - 1) // BLOCK_SIZE  # output elements
        )
        
        # Launch kernel
        conv3d_kernel[grid](
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
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
            dilation_d,
            dilation_h,
            dilation_w,
            self.groups,
            self.in_channels // self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        return output