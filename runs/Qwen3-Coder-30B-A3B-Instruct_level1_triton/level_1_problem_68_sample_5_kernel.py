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
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    output_padding_d,
    output_padding_w,
    output_padding_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_w = tl.program_id(3)
    out_h = tl.program_id(4)
    
    # Calculate group information
    group_idx = out_channel_idx // CHANNELS_PER_GROUP
    local_channel_idx = out_channel_idx % CHANNELS_PER_GROUP
    
    # Shared memory for input tiles
    tile_size = 16
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(tile_size, tile_size, tile_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for k_d in range(kernel_depth):
        for k_w in range(kernel_width):
            for k_h in range(kernel_height):
                # Calculate input coordinates
                d_in = out_d * stride_d - padding_d + k_d
                w_in = out_w * stride_w - padding_w + k_w
                h_in = out_h * stride_h - padding_h + k_h
                
                # Check bounds
                if (d_in >= 0 and d_in < input_depth and 
                    w_in >= 0 and w_in < input_width and 
                    h_in >= 0 and h_in < input_height):
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * input_depth * input_width * input_height) +
                                       local_channel_idx * (input_depth * input_width * input_height) +
                                       d_in * (input_width * input_height) +
                                       w_in * input_height +
                                       h_in)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_channel_idx * (in_channels * kernel_depth * kernel_width * kernel_height) +
                                        local_channel_idx * (kernel_depth * kernel_width * kernel_height) +
                                        k_d * (kernel_width * kernel_height) +
                                        k_w * kernel_height +
                                        k_h)
                    
                    acc += input_val * weight_val
    
    # Add bias if exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    if (out_d < output_depth and out_w < output_width and out_h < output_height):
        output_idx = batch_idx * (out_channels * output_depth * output_width * output_height) + \
                     out_channel_idx * (output_depth * output_width * output_height) + \
                     out_d * (output_width * output_height) + \
                     out_w * output_height + \
                     out_h
        tl.store(output_ptr + output_idx, acc)

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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_depth, input_width, input_height = x.shape
        kernel_depth, kernel_width, kernel_height = self.kernel_size
        stride_d, stride_w, stride_h = self.stride
        padding_d, padding_w, padding_h = self.padding
        output_padding_d, output_padding_w, output_padding_h = self.output_padding
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d
        output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w
        output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_width, output_height, device=x.device, dtype=torch.float32)
        
        # Prepare pointers for Triton kernel
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        output_ptr = output.data_ptr()
        
        # Configure grid dimensions
        grid = (
            batch_size,
            self.out_channels,
            output_depth,
            output_width,
            output_height
        )
        
        # Define kernel parameters
        BLOCK_SIZE = 16
        GROUPS = self.groups
        CHANNELS_PER_GROUP = self.out_channels // GROUPS
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_width,
            input_height,
            output_depth,
            output_width,
            output_height,
            kernel_depth,
            kernel_width,
            kernel_height,
            stride_d,
            stride_w,
            stride_h,
            padding_d,
            padding_w,
            padding_h,
            output_padding_d,
            output_padding_w,
            output_padding_h,
            GROUPS,
            BLOCK_SIZE,
            GROUPS,
            CHANNELS_PER_GROUP
        )
        
        return output