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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    input_depth,
    kernel_height,
    kernel_width,
    kernel_depth,
    output_height,
    output_width,
    output_depth,
    stride_h,
    stride_w,
    stride_d,
    padding_h,
    padding_w,
    padding_d,
    dilation_h,
    dilation_w,
    dilation_d,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    THREADS_PER_GROUP: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    output_w_idx = tl.program_id(3)
    output_d_idx = tl.program_id(4)
    
    # Calculate output dimensions
    output_size = output_height * output_width * output_depth
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate group information
    group_id = channel_idx // group_size
    weight_offset = group_id * out_channels * in_channels // groups * kernel_height * kernel_width * kernel_depth
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for kd in range(kernel_depth):
                # Calculate input coordinates
                ih = output_h_idx * stride_h + kh * dilation_h - padding_h
                iw = output_w_idx * stride_w + kw * dilation_w - padding_w
                id = output_d_idx * stride_d + kd * dilation_d - padding_d
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width and id >= 0 and id < input_depth:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * input_height * input_width * input_depth +
                                       channel_idx * input_height * input_width * input_depth +
                                       ih * input_width * input_depth +
                                       iw * input_depth +
                                       id)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        weight_offset + 
                                        channel_idx * out_channels * kernel_height * kernel_width * kernel_depth +
                                        (channel_idx % group_size) * out_channels * kernel_height * kernel_width * kernel_depth +
                                        kh * kernel_width * kernel_depth +
                                        kw * kernel_depth +
                                        kd)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Write output
    if batch_idx < batch_size and channel_idx < out_channels:
        output_offset = batch_idx * out_channels * output_height * output_width * output_depth + \
                       channel_idx * output_height * output_width * output_depth + \
                       output_h_idx * output_width * output_depth + \
                       output_w_idx * output_depth + \
                       output_d_idx
        tl.store(output_ptr + output_offset, acc[0])

class ModelNew(nn.Module):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        # Ensure input is on GPU and contiguous
        x = x.contiguous().cuda()
        
        # Get dimensions
        batch_size, in_channels, height, width, depth = x.shape
        out_channels, _, kernel_height, kernel_width, kernel_depth = self.weight.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - (self.dilation * (kernel_height - 1) + 1)) // self.stride + 1
        output_width = (width + 2 * self.padding - (self.dilation * (kernel_width - 1) + 1)) // self.stride + 1
        output_depth = (depth + 2 * self.padding - (self.dilation * (kernel_depth - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, out_channels, output_height, output_width, output_depth, device=x.device, dtype=torch.float32)
        
        # Define block size and threads per group
        BLOCK_SIZE = 16
        THREADS_PER_GROUP = 32
        
        # Grid configuration
        grid = (
            batch_size,
            out_channels,
            output_height,
            output_width,
            output_depth
        )
        
        # Launch kernel
        conv3d_kernel[grid](
            x,
            self.weight,
            output,
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            depth,
            kernel_height,
            kernel_width,
            kernel_depth,
            output_height,
            output_width,
            output_depth,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.dilation,
            self.groups,
            in_channels // self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            THREADS_PER_GROUP=THREADS_PER_GROUP
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output