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
    OUTPUT_SIZE_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output dimensions
    total_output_elements = output_height * output_width
    
    # Each thread block processes one output element
    output_offset = output_idx * OUTPUT_SIZE_PER_BLOCK
    if output_offset >= total_output_elements:
        return
        
    # Process multiple channels per block
    channel_offset = channel_idx * CHANNELS_PER_BLOCK
    if channel_offset >= out_channels:
        return
        
    # Shared memory for input tiles
    shared_input = tl.shared_pointer(input_ptr, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Loop over kernel
    for k in range(in_channels):
        # Load kernel weights for this channel
        weight_base = weight_ptr + k * out_channels * kernel_height * kernel_width
        out_base = output_ptr + batch_idx * out_channels * output_height * output_width
        
        # For each output position
        for out_y in range(output_height):
            for out_x in range(output_width):
                # Calculate corresponding input position
                input_y = out_y * stride_h - pad_h
                input_x = out_x * stride_w - pad_w
                
                acc = 0.0
                # Convolve with kernel
                for ky in range(kernel_height):
                    for kx in range(kernel_width):
                        input_pos_y = input_y + ky
                        input_pos_x = input_x + kx
                        
                        # Check bounds
                        if input_pos_y >= 0 and input_pos_y < input_height and \
                           input_pos_x >= 0 and input_pos_x < input_width:
                            input_val = tl.load(input_ptr + 
                                              batch_idx * in_channels * input_height * input_width +
                                              k * input_height * input_width +
                                              input_pos_y * input_width + 
                                              input_pos_x)
                            
                            weight_val = tl.load(weight_base + 
                                               k * out_channels * kernel_height * kernel_width +
                                               channel_offset * kernel_height * kernel_width +
                                               ky * kernel_width + kx)
                            acc += input_val * weight_val
                
                # Add bias if present
                if bias_ptr is not None:
                    bias_val = tl.load(bias_ptr + channel_offset)
                    acc += bias_val
                    
                # Store result
                output_pos = batch_idx * out_channels * output_height * output_width + \
                           channel_offset * output_height * output_width + \
                           out_y * output_width + out_x
                tl.store(out_ptr + output_pos, acc)

# Simplified fused implementation for better performance
@triton.jit
def fused_conv_transpose2d_kernel(
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
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr
):
    # Thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    tile_y = tl.program_id(2)
    tile_x = tl.program_id(3)
    
    # Tile coordinates
    start_y = tile_y * TILE_SIZE_H
    start_x = tile_x * TILE_SIZE_W
    
    # Shared memory for input tile
    input_tile = tl.shared_pointer(input_ptr, (TILE_SIZE_H + 2*pad_h, TILE_SIZE_W + 2*pad_w))
    
    # Process this tile
    for k in range(in_channels):
        # Compute output for this channel and tile
        acc = 0.0
        
        # Convolution loop
        for ky in range(kernel_height):
            for kx in range(kernel_width):
                # Calculate input position
                input_y = start_y * stride_h - pad_h + ky
                input_x = start_x * stride_w - pad_w + kx
                
                # Check bounds
                if input_y >= 0 and input_y < input_height and \
                   input_x >= 0 and input_x < input_width:
                    input_val = tl.load(input_ptr + 
                                      batch_idx * in_channels * input_height * input_width +
                                      k * input_height * input_width +
                                      input_y * input_width + 
                                      input_x)
                    
                    weight_val = tl.load(weight_ptr + 
                                       k * out_channels * kernel_height * kernel_width +
                                       out_ch_idx * kernel_height * kernel_width +
                                       ky * kernel_width + kx)
                    acc += input_val * weight_val
        
        # Add bias
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_ch_idx)
            acc += bias_val
            
        # Store output
        out_y = start_y
        out_x = start_x
        if out_y < output_height and out_x < output_width:
            output_pos = batch_idx * out_channels * output_height * output_width + \
                       out_ch_idx * output_height * output_width + \
                       out_y * output_width + out_x
            tl.store(output_ptr + output_pos, acc)

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
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Compute output dimensions
        output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
        output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, dtype=torch.float32, device=x.device)
        
        # Prepare input tensor for kernel (ensure it's contiguous)
        x_contiguous = x.contiguous()
        
        # Set up kernel launch parameters
        TILE_SIZE_H = 16
        TILE_SIZE_W = 16
        
        # Grid configuration
        grid = (
            batch_size,
            self.out_channels,
            (output_height + TILE_SIZE_H - 1) // TILE_SIZE_H,
            (output_width + TILE_SIZE_W - 1) // TILE_SIZE_W
        )
        
        # Launch kernel
        fused_conv_transpose2d_kernel[grid](
            x_contiguous,
            self.weight,
            output,
            self.bias_param,
            batch_size,
            self.in_channels,
            self.out_channels,
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
            TILE_SIZE_H=TILE_SIZE_H,
            TILE_SIZE_W=TILE_SIZE_W
        )
        
        return output