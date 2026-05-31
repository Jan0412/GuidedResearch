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
    input_size,
    output_size,
    kernel_size,
    in_channels,
    out_channels,
    stride,
    padding,
    dilation,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get program ID and create block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate grid dimensions
    grid_size = tl.cdiv(output_size * batch_size * out_channels, BLOCK_SIZE)
    
    # Shared memory for input and weight
    shared_input = tl.shared_pointer(input_ptr, BLOCK_SIZE)
    shared_weight = tl.shared_pointer(weight_ptr, BLOCK_SIZE)
    
    # Process elements in parallel
    for i in range(block_start, min(block_start + BLOCK_SIZE, grid_size)):
        # Compute indices
        batch_idx = i // (output_size * out_channels)
        out_ch_idx = (i // output_size) % out_channels
        out_pos_idx = i % output_size
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Convolution computation
        for k in range(kernel_size):
            # Compute input position
            input_pos = out_pos_idx * stride - padding + k * dilation
            
            # Check bounds
            if input_pos >= 0 and input_pos < input_size:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * input_size) +
                                  out_ch_idx * input_size + 
                                  input_pos, 
                                  mask=(input_pos < input_size))
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   out_ch_idx * (in_channels * kernel_size) +
                                   k * in_channels + 
                                   out_ch_idx, 
                                   mask=True)
                
                # Accumulate
                acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_size) +
                out_ch_idx * output_size + 
                out_pos_idx, 
                acc)

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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        batch_size, in_channels, input_length = x.shape
        
        # Compute output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) * self.dilation + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Call Triton kernel
        if output_length > 0:
            self._triton_conv1d_transpose(x, self.weight, output, self.bias)
        
        return output
    
    def _triton_conv1d_transpose(self, x, weight, output, bias):
        """Internal method to run the Triton kernel"""
        batch_size, in_channels, input_length = x.shape
        output_length = output.shape[2]
        
        # Flatten tensors for easier indexing
        x_flat = x.contiguous().view(-1)
        weight_flat = weight.contiguous().view(-1)
        output_flat = output.contiguous().view(-1)
        
        # Setup kernel launch parameters
        BLOCK_SIZE = 1024
        grid_size = math.ceil(output_length * batch_size * self.out_channels / BLOCK_SIZE)
        
        # Launch kernel
        conv1d_transpose_kernel[grid_size](
            x_flat,
            weight_flat,
            output_flat,
            input_length,
            output_length,
            self.kernel_size,
            in_channels,
            self.out_channels,
            self.stride,
            self.padding,
            self.dilation,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE_M=8
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)