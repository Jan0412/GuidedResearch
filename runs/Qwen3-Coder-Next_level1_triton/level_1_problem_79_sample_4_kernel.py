import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, kernel_size,
    input_length, output_length,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output element
    # We use 2D grid: one dimension for output position, one for batch/channel combination
    batch_channel_idx = tl.program_id(0)
    out_pos = tl.program_id(1)
    
    # Decode batch and channel indices
    batch_idx = batch_channel_idx // out_channels
    channel_idx = batch_channel_idx % out_channels
    
    # Bounds check
    if batch_idx >= batch_size or out_pos >= output_length:
        return
    
    # Compute the starting input position for this output position
    # For transposed convolution: out_pos = batch_idx * out_channels * output_length + channel_idx * output_length + out_pos
    # The relationship is: out_pos = input_pos * stride + kernel_pos * dilation - padding
    # Solving for input_pos: input_pos = (out_pos + padding - kernel_pos * dilation) / stride
    
    # Accumulate contributions from all kernel positions
    acc = 0.0
    
    # Loop over kernel positions
    for k in range(kernel_size):
        # Calculate the input position that contributes to this output position
        # For transposed convolution with given kernel position k:
        # out_pos = input_pos * stride + k * dilation - padding
        # => input_pos = (out_pos + padding - k * dilation) / stride
        
        input_pos = (out_pos + padding - k * dilation) // stride
        
        # Check if this input position is valid
        if input_pos * stride == (out_pos + padding - k * dilation) and 0 <= input_pos < input_length:
            # Calculate indices
            x_offset = batch_idx * in_channels * input_length + channel_idx * input_length + input_pos
            w_offset = channel_idx * out_channels * kernel_size + k * out_channels + channel_idx  # Wrong indexing, fix below
            
            # Actually, weight indexing should be: [in_channel, out_channel, kernel_pos]
            # Since we're processing a specific output channel, we need to loop over input channels
            pass
    
    # Redesign: process one output element per program, but loop over input channels and kernel positions
    # Let's restructure: each program handles one (batch, out_channel, out_pos)
    
    # Reset for correct indexing
    batch_idx = tl.program_id(0) // out_channels
    out_channel_idx = tl.program_id(0) % out_channels
    
    if batch_idx >= batch_size or out_pos >= output_length:
        return
    
    # Accumulate over input channels and kernel positions
    acc = 0.0
    
    # For each input channel
    for in_ch in range(in_channels):
        # For each kernel position
        for k in range(kernel_size):
            # Calculate input position that contributes to this output
            # out_pos = input_pos * stride + k * dilation - padding
            # => input_pos = (out_pos + padding - k * dilation) / stride
            numerator = out_pos + padding - k * dilation
            if numerator % stride == 0:
                input_pos = numerator // stride
                if 0 <= input_pos < input_length:
                    # Calculate pointers
                    x_offset = batch_idx * in_channels * input_length + in_ch * input_length + input_pos
                    w_offset = in_ch * out_channels * kernel_size + out_channel_idx * kernel_size + k
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_offset)
                    w_val = tl.load(w_ptr + w_offset)
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = out_channel_idx
        acc += tl.load(b_ptr + b_offset)
    
    # Store result
    out_offset = batch_idx * out_channels * output_length + out_channel_idx * output_length + out_pos
    tl.store(out_ptr + out_offset, acc)


def triton_conv1d_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    
    # Calculate output length for transposed convolution
    # out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_channels
    # Actually: out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # More accurately: out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # Standard formula: out_length = (input_length - 1) * stride - 2 * padding + kernel_size + padding_out
    # For transposed convolution: out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # Simplified: out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # Let me use the correct formula: 
    # For ConvTranspose1d: output_padding = 0, so
    # out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # With output_padding=0: out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Grid configuration: (batch_size * out_channels, output_length)
    grid = (batch_size * out_channels, output_length)
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        input_length, output_length,
        stride, padding, dilation,
        BLOCK_SIZE=128
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight and bias as in the original nn.ConvTranspose1d
        # Weight shape: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weight and bias parameters similar to PyTorch's ConvTranspose1d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom Triton kernel.
        """
        return triton_conv1d_transpose(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )


import math