import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_SIZE_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output position
    output_h = output_idx // output_width
    output_w = output_idx % output_width
    
    # Check bounds
    if output_h >= output_height or output_w >= output_width:
        return
        
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            input_h = output_h * stride_h + kh - pad_h
            input_w = output_w * stride_w + kw - pad_w
            
            # Check if input position is valid
            if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * input_height * input_width) +
                                  channel_idx * (input_height * input_width) +
                                  input_h * input_width + input_w)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_idx * (out_channels * kernel_height * kernel_width) +
                                   output_idx * (kernel_height * kernel_width) +
                                   kh * kernel_width + kw)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + output_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * (out_channels * output_height * output_width) +
             output_idx * (output_height * output_width) +
             output_h * output_width + output_w, 
             acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        output_height * output_width
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        BLOCK_SIZE=32,
        CHANNELS_PER_BLOCK=16,
        OUTPUT_SIZE_PER_BLOCK=32
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        # Use the Triton implementation
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)