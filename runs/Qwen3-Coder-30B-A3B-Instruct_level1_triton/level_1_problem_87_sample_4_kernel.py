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
    IN_CHANNELS_BLOCK: tl.constexpr,
    OUT_CHANNELS_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    h_id = tl.program_id(2)
    w_id = tl.program_id(3)
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * height * width
    out_ch_offset = out_ch_id * height * width
    
    # Shared memory for input channel accumulation
    acc = tl.zeros((OUT_CHANNELS_BLOCK,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for in_ch_block in range(0, in_channels, IN_CHANNELS_BLOCK):
        # Load weights for this output channel and input channel block
        weight_offsets = out_ch_id * in_channels + in_ch_block
        weight_ptrs = weight_ptr + weight_offsets
        
        # Load input data for current batch, height, width position
        input_offset = batch_offset + in_ch_block * height * width + h_id * width + w_id
        input_ptrs = input_ptr + input_offset
        
        # Process input channels in this block
        for i in range(IN_CHANNELS_BLOCK):
            if in_ch_block + i < in_channels:
                # Load weight and input value
                weight_val = tl.load(weight_ptrs + i, mask=(in_ch_block + i < in_channels))
                input_val = tl.load(input_ptrs + i * height * width, mask=True)
                # Accumulate
                acc += weight_val * input_val
    
    # Write output
    if out_ch_id < out_channels:
        output_offset = batch_id * out_channels * height * width + out_ch_offset + h_id * width + w_id
        output_ptr += output_offset
        tl.store(output_ptr, acc[0], mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, height, width, device=x.device, dtype=torch.float32)
        
        # Define kernel launch parameters
        BLOCK_SIZE = 1024
        IN_CHANNELS_BLOCK = 32
        OUT_CHANNELS_BLOCK = 1
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            (self.out_channels + OUT_CHANNELS_BLOCK - 1) // OUT_CHANNELS_BLOCK,
            height,
            width
        )
        
        # Launch kernel
        pointwise_conv2d_kernel[grid](
            x,
            self.weight,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            height,
            width,
            BLOCK_SIZE=BLOCK_SIZE,
            IN_CHANNELS_BLOCK=IN_CHANNELS_BLOCK,
            OUT_CHANNELS_BLOCK=OUT_CHANNELS_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output