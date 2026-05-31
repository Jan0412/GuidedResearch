import torch
import torch.nn as nn
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
    OUTPUT_H_PER_BLOCK: tl.constexpr,
    OUTPUT_W_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_block = tl.program_id(1)
    output_h_block = tl.program_id(2)
    output_w_block = tl.program_id(3)
    
    # Calculate output positions
    output_h_start = output_h_block * OUTPUT_H_PER_BLOCK
    output_w_start = output_w_block * OUTPUT_W_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_H_PER_BLOCK + 2 * pad_h, OUTPUT_W_PER_BLOCK + 2 * pad_w))
    
    # Process multiple channels per block
    for c in range(channel_block * CHANNELS_PER_BLOCK, min((channel_block + 1) * CHANNELS_PER_BLOCK, in_channels)):
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_H_PER_BLOCK, OUTPUT_W_PER_BLOCK), dtype=tl.float32)
        
        # Convolve over kernel
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input positions
                input_h_start = output_h_start * stride_h - pad_h + kh
                input_w_start = output_w_start * stride_w - pad_w + kw
                
                # Load input data (with padding handling)
                for oh in range(OUTPUT_H_PER_BLOCK):
                    for ow in range(OUTPUT_W_PER_BLOCK):
                        h = input_h_start + oh
                        w = input_w_start + ow
                        
                        if h >= 0 and h < input_height and w >= 0 and w < input_width:
                            input_val = tl.load(input_ptr + 
                                              batch_idx * (in_channels * input_height * input_width) +
                                              c * (input_height * input_width) +
                                              h * input_width + w)
                        else:
                            input_val = 0.0
                        
                        # Load weight
                        weight_val = tl.load(weight_ptr + 
                                           c * (out_channels * kernel_height * kernel_width) +
                                           tl.program_id(4) * (kernel_height * kernel_width) +
                                           kh * kernel_width + kw)
                        
                        acc[oh, ow] += input_val * weight_val
        
        # Apply bias if available
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + tl.program_id(4))
            for oh in range(OUTPUT_H_PER_BLOCK):
                for ow in range(OUTPUT_W_PER_BLOCK):
                    acc[oh, ow] += bias_val
        
        # Store output
        for oh in range(OUTPUT_H_PER_BLOCK):
            for ow in range(OUTPUT_W_PER_BLOCK):
                if output_h_start + oh < output_height and output_w_start + ow < output_width:
                    tl.store(output_ptr + 
                           batch_idx * (out_channels * output_height * output_width) +
                           tl.program_id(4) * (output_height * output_width) +
                           (output_h_start + oh) * output_width + (output_w_start + ow),
                           acc[oh, ow])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton implementation of 2D transposed convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 4
    OUTPUT_H_PER_BLOCK = 8
    OUTPUT_W_PER_BLOCK = 8
    
    # Grid dimensions
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (output_height + OUTPUT_H_PER_BLOCK - 1) // OUTPUT_H_PER_BLOCK,
        (output_width + OUTPUT_W_PER_BLOCK - 1) // OUTPUT_W_PER_BLOCK,
        out_channels  # For channel dimension
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_H_PER_BLOCK=OUTPUT_H_PER_BLOCK,
        OUTPUT_W_PER_BLOCK=OUTPUT_W_PER_BLOCK
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
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )