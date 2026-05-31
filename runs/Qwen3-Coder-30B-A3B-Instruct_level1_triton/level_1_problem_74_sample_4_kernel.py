import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_transpose_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Shared memory for weight tiles
    tile_weight = tl.shared_ptr(weight_ptr, shape=(GROUP_SIZE, kernel_size), dtype=tl.float32)
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for ch in range(in_channels):
        # Load input slice
        input_slice = input_ptr + batch_idx * in_channels * input_length + ch * input_length
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate output position
            output_pos = k * dilation - padding
            
            # Calculate input position
            input_pos = output_pos // stride
            
            # Check if this is valid
            if input_pos >= 0 and input_pos < input_length and output_pos % stride == 0:
                # Load weight
                weight_val = tl.load(weight_ptr + out_ch_idx * in_channels * kernel_size + ch * kernel_size + k)
                
                # Load input value
                input_val = tl.load(input_slice + input_pos, mask=(input_pos < input_length))
                
                # Accumulate
                acc += weight_val * input_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Write output
    output_slice = output_ptr + batch_idx * out_channels * output_length + out_ch_idx * output_length
    for i in range(BLOCK_SIZE):
        output_pos = tl.program_id(2) * BLOCK_SIZE + i
        if output_pos < output_length:
            tl.store(output_slice + output_pos, acc[i])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + (self.dilation * (self.kernel_size - 1) + 1)
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Prepare input tensor for kernel (make contiguous)
        x = x.contiguous()
        
        # Define block size and group size
        BLOCK_SIZE = 256
        GROUP_SIZE = 32
        
        # Calculate grid dimensions
        grid_batch = batch_size
        grid_out_ch = self.out_channels
        grid_output = (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        conv1d_transpose_kernel[
            (grid_batch, grid_out_ch, grid_output),
            num_warps=4,
            num_stages=3
        ](
            x,
            self.weight,
            output,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        return output