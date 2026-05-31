import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output width index
    output_w_idx = tl.program_id(3)
    
    # Ensure we're within valid bounds
    if output_h_idx >= output_height or output_w_idx >= output_width:
        return
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate input positions
    input_start_h = output_h_idx * stride_h - padding_h
    input_start_w = output_w_idx * stride_w - padding_w
    
    # Iterate over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate actual input position with dilation
            input_h = input_start_h + kh * dilation_h
            input_w = input_start_w + kw * dilation_w
            
            # Check bounds
            if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                # Load input value
                input_val = tl.load(input_ptr + 
                    batch_idx * (in_channels * input_height * input_width) +
                    channel_idx * (input_height * input_width) +
                    input_h * input_width + input_w)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                    channel_idx * (kernel_height * kernel_width) +
                    kh * kernel_width + kw)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    output_idx = batch_idx * (in_channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_h_idx * output_width + output_w_idx
    
    tl.store(output_ptr + output_idx, acc[0])


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        batch_size, in_channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding_h - (self.dilation_h * (self.kernel_size_h - 1) + 1)) // self.stride_h + 1
        output_width = (input_width + 2 * self.padding_w - (self.dilation_w * (self.kernel_size_w - 1) + 1)) // self.stride_w + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Grid configuration
        grid = (
            batch_size,
            in_channels,
            output_height,
            output_width
        )
        
        # Launch kernel
        BLOCK_SIZE = 16
        CHANNELS_PER_BLOCK = 1
        
        depthwise_conv2d_kernel[grid](
            x,
            weight,
            output,
            batch_size,
            in_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            self.kernel_size_h,
            self.kernel_size_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output