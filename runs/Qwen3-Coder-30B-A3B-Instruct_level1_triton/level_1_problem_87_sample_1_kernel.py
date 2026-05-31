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
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    h_id = tl.program_id(2)
    w_id = tl.program_id(3)
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * height * width
    out_channel_offset = out_channel_id * height * width
    
    # Shared memory for input channel accumulation
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(IN_CHANNELS_BLOCK_SIZE,))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(IN_CHANNELS_BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process in chunks of IN_CHANNELS_BLOCK_SIZE
    for i in range(0, in_channels, IN_CHANNELS_BLOCK_SIZE):
        # Load input and weight data
        input_idx = batch_offset + i * height * width + h_id * width + w_id
        weight_idx = out_channel_id * in_channels + i
        
        # Load input chunk
        input_chunk = tl.load(input_ptr + input_idx, mask=(i + tl.arange(0, IN_CHANNELS_BLOCK_SIZE)) < in_channels, other=0.0)
        # Load weight chunk
        weight_chunk = tl.load(weight_ptr + weight_idx, mask=(i + tl.arange(0, IN_CHANNELS_BLOCK_SIZE)) < in_channels, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(input_chunk * weight_chunk)
    
    # Store result
    if out_channel_id < out_channels:
        output_idx = batch_id * out_channels * height * width + out_channel_offset + h_id * width + w_id
        tl.store(output_ptr + output_idx, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size, _, height, width = x.shape
        
        # Ensure inputs are contiguous and on GPU
        x = x.contiguous().to(torch.float32)
        weight = self.weight.to(torch.float32)
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, height, width, device=x.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_SIZE = 1024
        IN_CHANNELS_BLOCK_SIZE = 32
        OUT_CHANNELS_BLOCK_SIZE = 32
        
        # Grid dimensions
        grid = (
            batch_size,
            (self.out_channels + OUT_CHANNELS_BLOCK_SIZE - 1) // OUT_CHANNELS_BLOCK_SIZE,
            height,
            width
        )
        
        # Launch kernel
        pointwise_conv2d_kernel[grid](
            x,
            weight,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            height,
            width,
            BLOCK_SIZE=BLOCK_SIZE,
            IN_CHANNELS_BLOCK_SIZE=IN_CHANNELS_BLOCK_SIZE,
            OUT_CHANNELS_BLOCK_SIZE=OUT_CHANNELS_BLOCK_SIZE
        )
        
        # Add bias if needed
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output

# For the test case provided in the original code
def get_inputs():
    batch_size = 16
    in_channels = 64
    out_channels = 128
    width = 1024
    height = 1024
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 128]