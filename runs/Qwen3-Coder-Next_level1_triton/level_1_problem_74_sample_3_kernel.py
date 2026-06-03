import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Calculate output position
    out_pos = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_pos < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for in_channel in range(in_channels):
        # Calculate input position for each output position
        # For transposed conv: out_pos = input_pos * stride + (kernel_pos - padding)
        # So input_pos = (out_pos - (kernel_pos - padding)) / stride
        
        # We need to iterate over kernel positions
        for kernel_pos in range(kernel_size):
            # Calculate corresponding input position
            # out_pos = input_pos * stride + kernel_pos * dilation - padding
            # input_pos = (out_pos + padding - kernel_pos * dilation) / stride
            input_pos = (out_pos + padding - kernel_pos * dilation) // stride
            
            # Check if input_pos is within bounds
            valid = (input_pos >= 0) & (input_pos < length) & ((out_pos + padding - kernel_pos * dilation) % stride == 0)
            
            # Load input values
            x_offset = batch_id * in_channels * length + in_channel * length + input_pos
            x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
            
            # Load weight values
            w_offset = in_channel * out_channels * kernel_size + out_channel_id * kernel_size + kernel_pos
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += tl.where(valid, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = out_channel_id
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = batch_id * out_channels * out_length + out_channel_id * out_length + out_pos
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of ConvTranspose1d
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation: convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    # For transposed convolution: out_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
    # Assuming out_padding=0 (default)
    out_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate grid dimensions
    grid = (batch_size, out_channels, (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


# Import math for sqrt in parameter initialization
import math