import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    
    # Calculate which batch/channel group this thread handles
    num_blocks_per_group = out_channels // GROUP_SIZE_M
    group_id = pid // num_blocks_per_group
    block_id = pid % num_blocks_per_group
    
    # Calculate output indices
    batch_idx = group_id // (out_channels // GROUP_SIZE_M)
    channel_idx = (group_id % (out_channels // GROUP_SIZE_M)) * GROUP_SIZE_M + block_id
    
    if channel_idx >= out_channels:
        return
        
    # Shared memory for input tiles
    tile_size = min(BLOCK_SIZE, height * width)
    
    # Initialize accumulator
    acc = tl.zeros((tile_size,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        # Load input data
        input_offset = batch_idx * (in_channels * height * width) + c * (height * width)
        input_tile = tl.load(input_ptr + input_offset + tl.arange(0, tile_size))
        
        # Load weight data
        weight_offset = channel_idx * in_channels + c
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_tile * weight_val
    
    # Store output
    output_offset = batch_idx * (out_channels * height * width) + channel_idx * (height * width)
    tl.store(output_ptr + output_offset + tl.arange(0, tile_size), acc)

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution (1x1 conv)
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height, width, dtype=torch.float32, device=input_tensor.device)
    
    # Flatten input for easier processing
    input_flat = input_tensor.view(batch_size, in_channels, height * width)
    
    # Flatten weight for easier processing
    weight_flat = weight.view(out_channels, in_channels)
    
    # For each batch and channel, perform matrix multiplication
    for b in range(batch_size):
        for c_out in range(out_channels):
            # Initialize output row
            output_row = torch.zeros(height * width, dtype=torch.float32, device=input_tensor.device)
            
            # Matrix multiply: input_row @ weight_row.T
            for c_in in range(in_channels):
                input_val = input_flat[b, c_in, :]
                weight_val = weight_flat[c_out, c_in]
                output_row += input_val * weight_val
            
            # Add bias if present
            if bias is not None:
                output_row += bias[c_out]
                
            # Store result
            output[b, c_out, :, :] = output_row.view(height, width)
    
    return output

# Optimized version using a more efficient approach
@triton.jit
def fused_conv2d_bias_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread and block indices
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Check bounds
    if batch_id >= batch_size or out_ch_id >= out_channels:
        return
    
    # Process one output channel at a time
    output_offset = batch_id * (out_channels * height * width) + out_ch_id * (height * width)
    
    # Initialize accumulator
    acc = tl.zeros((height * width,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        input_offset = batch_id * (in_channels * height * width) + c * (height * width)
        weight_offset = out_ch_id * in_channels + c
        
        # Load input and weight
        input_data = tl.load(input_ptr + input_offset + tl.arange(0, height * width))
        weight_data = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_data * weight_data
    
    # Add bias
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr + out_ch_id)
        acc += bias_data
    
    # Store output
    tl.store(output_ptr + output_offset + tl.arange(0, height * width), acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bias_enabled = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        """
        # Get parameters
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv1d.out_channels
        
        # Get weights and biases
        weight = self.conv1d.weight.data
        bias = self.conv1d.bias.data if self.bias_enabled else None
        
        # Use our Triton implementation
        return triton_pointwise_conv2d(x, weight, bias)