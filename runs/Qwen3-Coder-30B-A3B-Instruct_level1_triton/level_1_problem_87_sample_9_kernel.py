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
    IN_CHANNELS_BLOCK_SIZE: tl.constexpr,
    OUT_CHANNELS_BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    height_idx = tl.program_id(1)
    width_idx = tl.program_id(2)
    
    # Shared memory for weight tiles
    tile_weight = tl.shared_tensor(tl.block_dim(0), tl.block_dim(1))
    
    # Initialize output accumulator
    acc = tl.zeros((OUT_CHANNELS_BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels in chunks
    for ic_start in range(0, in_channels, IN_CHANNELS_BLOCK_SIZE):
        # Load input for this batch, height, width, and channel chunk
        input_chunk = tl.load(input_ptr + 
                             batch_idx * (in_channels * height * width) +
                             ic_start * (height * width) +
                             height_idx * width +
                             width_idx,
                             mask=(ic_start + tl.arange(0, IN_CHANNELS_BLOCK_SIZE) < in_channels),
                             other=0.0)
        
        # Load weight chunk for this channel and output channel
        weight_chunk = tl.load(weight_ptr + 
                              tl.arange(0, OUT_CHANNELS_BLOCK_SIZE)[:, None] * in_channels +
                              ic_start + tl.arange(0, IN_CHANNELS_BLOCK_SIZE)[None, :],
                              mask=(ic_start + tl.arange(0, IN_CHANNELS_BLOCK_SIZE) < in_channels) &
                                   (tl.arange(0, OUT_CHANNELS_BLOCK_SIZE)[:, None] < out_channels),
                              other=0.0)
        
        # Perform dot product
        acc += tl.sum(input_chunk[None, :] * weight_chunk, axis=1)
    
    # Store output
    tl.store(output_ptr + 
             batch_idx * (out_channels * height * width) +
             tl.arange(0, OUT_CHANNELS_BLOCK_SIZE)[:, None, None] * (height * width) +
             height_idx * width +
             width_idx,
             acc[:, None, None])

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution.
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 128
    IN_CHANNELS_BLOCK_SIZE = 32
    OUT_CHANNELS_BLOCK_SIZE = 32
    
    # Grid dimensions
    grid = (
        batch_size,
        height,
        width
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
        IN_CHANNELS_BLOCK_SIZE=IN_CHANNELS_BLOCK_SIZE,
        OUT_CHANNELS_BLOCK_SIZE=OUT_CHANNELS_BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using Triton optimization.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Extract parameters
        weight = self.conv1d.weight
        bias = self.conv1d.bias if self.bias else None
        
        # Call Triton kernel
        return triton_pointwise_conv2d(x, weight, bias)