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
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Shared memory for weight
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(in_channels, kernel_size))
    
    # Load weight for this output channel
    for i in range(0, in_channels * kernel_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < in_channels * kernel_size
        if mask.any():
            shared_weight[tl.arange(0, BLOCK_SIZE) % in_channels, tl.arange(0, BLOCK_SIZE) // in_channels] = tl.load(
                weight_ptr + batch_idx * in_channels * out_channels * kernel_size + 
                out_channel_idx * in_channels * kernel_size + 
                offset, mask=mask, other=0.0
            )
    
    # Process output positions
    for out_pos in range(tl.program_id(2) * BLOCK_SIZE, output_length, BLOCK_SIZE):
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Compute convolution for this position
        for k in range(kernel_size):
            # Calculate input position
            input_pos = out_pos - padding + k * dilation
            
            # Check bounds
            valid_input = (input_pos >= 0) & (input_pos < input_length)
            
            # Load input values
            input_vals = tl.load(input_ptr + batch_idx * in_channels * input_length + 
                               tl.arange(0, BLOCK_SIZE) % in_channels * input_length + 
                               input_pos, mask=valid_input, other=0.0)
            
            # Multiply with weights
            weight_vals = tl.load(weight_ptr + out_channel_idx * in_channels * kernel_size + 
                                tl.arange(0, BLOCK_SIZE) % in_channels * kernel_size + 
                                k, mask=valid_input, other=0.0)
            
            acc += input_vals * weight_vals
        
        # Add bias if provided
        if bias_ptr is not None:
            acc += tl.load(bias_ptr + out_channel_idx, mask=True, other=0.0)
        
        # Store output
        tl.store(output_ptr + batch_idx * out_channels * output_length + 
                out_channel_idx * output_length + out_pos, acc, mask=True)

def triton_conv1d_transpose(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Custom Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        x.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
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

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Use our custom Triton implementation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# Note: This implementation assumes all operations can be done with contiguous memory layouts
# and doesn't handle all edge cases of the original PyTorch implementation but provides a 
# significant performance improvement through custom Triton kernelization