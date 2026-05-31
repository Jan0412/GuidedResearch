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
    height,
    width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    tile_size = kernel_height * kernel_width
    input_tile = tl.shared_ptr(tl.float32, tile_size)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load weights
    weight = tl.load(weight_ptr + channel_idx * kernel_height * kernel_width)
    
    # Calculate input indices
    input_h_start = output_h_idx * stride_h - padding_h
    input_w_start = output_w_idx * stride_w - padding_w
    
    # Perform convolution
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = input_h_start + kh * dilation_h
            iw = input_w_start + kw * dilation_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * in_channels * height * width +
                                  channel_idx * height * width +
                                  ih * width + iw)
                acc += input_val * weight
    
    # Store result
    if output_h_idx < output_height and output_w_idx < output_width:
        tl.store(output_ptr + 
                batch_idx * in_channels * output_height * output_width +
                channel_idx * output_height * output_width +
                output_h_idx * output_width + output_w_idx,
                acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        kernel_height = self.kernel_size
        kernel_width = 1
        stride_h = self.stride
        stride_w = self.stride
        padding_h = self.padding
        padding_w = self.padding
        dilation_h = self.dilation
        dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (height + 2 * padding_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
        output_width = (width + 2 * padding_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, in_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Launch kernel
        grid = (
            batch_size,
            in_channels,
            output_height,
            output_width
        )
        
        # Define block size
        BLOCK_SIZE = 1024
        CHANNELS_PER_BLOCK = 1
        
        # Launch kernel with appropriate grid
        depthwise_conv2d_kernel[grid](
            x,
            weight,
            output,
            batch_size,
            in_channels,
            height,
            width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_height,
            output_width,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
        
        # Add bias if needed
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output

    def extra_repr(self):
        return f'in_channels={self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias}'