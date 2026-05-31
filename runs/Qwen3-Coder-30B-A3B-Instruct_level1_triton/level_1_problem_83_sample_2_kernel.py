import torch
import torch.nn as nn
import triton
import triton.language as tl

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
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_w_idx = tl.program_id(3)
    
    # Ensure we don't go out of bounds
    if output_h_idx >= output_height or output_w_idx >= output_width:
        return
    
    # Calculate input positions
    input_h_start = output_h_idx * stride_h - padding_h
    input_w_start = output_w_idx * stride_w - padding_w
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            ih = input_h_start + kh * dilation_h
            iw = input_w_start + kw * dilation_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * in_channels * height * width +
                                  channel_idx * height * width +
                                  ih * width + iw)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_idx * kernel_height * kernel_width +
                                   kh * kernel_width + kw)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Write result
    tl.store(output_ptr + 
             batch_idx * in_channels * output_height * output_width +
             channel_idx * output_height * output_width +
             output_h_idx * output_width + output_w_idx,
             acc[0])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        output_width = (width + 2 * self.padding - (self.dilation * (1 - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=x.dtype)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare kernel parameters
        kernel_height = self.kernel_size
        kernel_width = 1
        
        # Launch kernel
        grid = (
            batch_size,
            channels,
            output_height,
            output_width
        )
        
        BLOCK_SIZE = 16
        
        # Call kernel
        depthwise_conv2d_kernel[grid](
            x,
            self.weight,
            output,
            batch_size,
            channels,
            height,
            width,
            kernel_height,
            kernel_width,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            output_height,
            output_width,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output