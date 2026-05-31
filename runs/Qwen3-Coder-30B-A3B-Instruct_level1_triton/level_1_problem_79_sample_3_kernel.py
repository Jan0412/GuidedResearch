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
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Shared memory for weight
    w_shared = tl.shared_pointer(weight_ptr, (out_channels, in_channels, kernel_size))
    
    # Load weight for this output channel
    w_row = tl.load(w_shared + out_ch_idx * in_channels * kernel_size, mask=tl.arange(0, kernel_size) < kernel_size)
    
    # Loop over input channels
    for in_ch_idx in range(in_channels):
        # Load input for this batch and channel
        input_row = tl.load(input_ptr + batch_idx * in_channels * input_size + in_ch_idx * input_size + tl.arange(0, input_size), mask=tl.arange(0, input_size) < input_size)
        
        # Compute output for this channel and output channel
        for i in range(output_size):
            acc = 0.0
            for k in range(kernel_size):
                # Calculate input position considering stride, padding, and dilation
                input_pos = i * stride - padding + k * dilation
                if 0 <= input_pos < input_size:
                    acc += input_row[input_pos] * w_row[k]
            
            # Store result
            tl.store(output_ptr + batch_idx * out_channels * output_size + out_ch_idx * output_size + i, acc)

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
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Launch kernel
        grid = (batch_size, self.out_channels)
        BLOCK_SIZE = 128
        GROUP_SIZE = 32
        
        conv1d_transpose_kernel[grid](
            x,
            weight,
            output,
            input_length,
            output_length,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
            
        return output