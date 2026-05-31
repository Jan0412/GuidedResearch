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
    OUTPUT_BLOCK_SIZE_H: tl.constexpr,
    OUTPUT_BLOCK_SIZE_W: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    channel_idx = tl.program_id(3)
    
    # Calculate output dimensions
    output_block_start_h = out_h_idx * OUTPUT_BLOCK_SIZE_H
    output_block_start_w = out_w_idx * OUTPUT_BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_BLOCK_SIZE_H + 2 * pad_h, OUTPUT_BLOCK_SIZE_W + 2 * pad_w))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_BLOCK_SIZE_H, OUTPUT_BLOCK_SIZE_W), dtype=tl.float32)
    
    # Process each channel
    for c in range(0, in_channels, CHANNELS_BLOCK_SIZE):
        # Load weights for this channel block
        weight_tile = tl.load(weight_ptr + 
                             (channel_idx * in_channels + c) * kernel_height * kernel_width +
                             tl.arange(0, kernel_height)[:, None] * kernel_width + 
                             tl.arange(0, kernel_width)[None, :])
        
        # Load input region for this block
        input_region_h_start = output_block_start_h * stride_h - pad_h
        input_region_w_start = output_block_start_w * stride_w - pad_w
        
        # Load input tile into shared memory
        for h in range(OUTPUT_BLOCK_SIZE_H + 2 * pad_h):
            for w in range(OUTPUT_BLOCK_SIZE_W + 2 * pad_w):
                ih = input_region_h_start + h
                iw = input_region_w_start + w
                
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    val = tl.load(input_ptr + 
                                 (batch_idx * in_channels + c) * input_height * input_width + 
                                 ih * input_width + iw)
                else:
                    val = 0.0
                
                shared_input[h, w] = val
        
        # Compute convolution for this channel block
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Load input from shared memory
                input_tile = shared_input[kh:kh+OUTPUT_BLOCK_SIZE_H, kw:kw+OUTPUT_BLOCK_SIZE_W]
                
                # Apply weight and accumulate
                weight_val = weight_tile[kh, kw]
                acc += input_tile * weight_val
    
    # Store result
    for h in range(OUTPUT_BLOCK_SIZE_H):
        for w in range(OUTPUT_BLOCK_SIZE_W):
            if output_block_start_h + h < output_height and output_block_start_w + w < output_width:
                output_idx = (batch_idx * out_channels + channel_idx) * output_height * output_width + \
                           (output_block_start_h + h) * output_width + (output_block_start_w + w)
                tl.store(output_ptr + output_idx, acc[h, w])

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding):
    """
    Custom Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid dimensions
    grid = (
        batch_size,
        math.ceil(output_height / 16),
        math.ceil(output_width / 16),
        out_channels
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
        BLOCK_SIZE=1024,
        OUTPUT_BLOCK_SIZE_H=16,
        OUTPUT_BLOCK_SIZE_W=16,
        CHANNELS_BLOCK_SIZE=32
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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)