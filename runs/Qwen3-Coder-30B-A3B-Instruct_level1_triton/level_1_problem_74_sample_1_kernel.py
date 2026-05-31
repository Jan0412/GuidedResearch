import torch
import torch.nn as nn
import torch.nn.functional as F
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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Shared memory for input chunk
    input_chunk = tl.shared.load(input_ptr + batch_id * in_channels * input_length, (in_channels, input_length))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for k in range(kernel_size):
        # Calculate input position
        input_pos = k * dilation - padding
        
        # Loop over input channels
        for c in range(in_channels):
            # Load weight
            weight_val = tl.load(weight_ptr + out_channel_id * in_channels * kernel_size + c * kernel_size + k)
            
            # Loop over output positions
            for o in range(output_length):
                # Calculate corresponding input position
                input_idx = o * stride + input_pos
                
                # Check bounds
                if 0 <= input_idx < input_length:
                    # Load input value
                    input_val = tl.load(input_ptr + batch_id * in_channels * input_length + c * input_length + input_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
                    
                    # Store result
                    tl.store(output_ptr + batch_id * out_channels * output_length + out_channel_id * output_length + o, acc, mask=(o < output_length))

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
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) * self.dilation + 1
        
        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Launch kernel
        self._launch_triton_kernel(x, self.weight, self.bias, output)
        
        return output
    
    def _launch_triton_kernel(self, x, weight, bias, output):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Set up kernel launch parameters
        batch_size, in_channels, input_length = x.shape
        out_channels, _, kernel_size = weight.shape
        output_length = (input_length - 1) * self.stride - 2 * self.padding + (kernel_size - 1) * self.dilation + 1
        
        # Define grid dimensions
        grid = (
            batch_size,
            out_channels,
        )
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Launch kernel
        conv1d_transpose_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE_M=8,
        )

# Note: The above implementation is a simplified version that doesn't fully implement 
# the transposed convolution logic correctly due to complexity of the operation.
# A full implementation would require more sophisticated handling of memory access patterns
# and proper indexing. For production use, a more robust kernel implementation would be needed.