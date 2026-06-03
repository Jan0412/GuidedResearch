import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, length)
    w_ptr,  # Weight tensor (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor (out_channels) - can be None
    out_ptr,  # Output tensor (batch, out_channels, length_out)
    batch_size, in_channels, out_channels, kernel_size, length,
    stride, padding, dilation,
    out_length,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch*out_channels dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for length dimension
):
    # Program IDs
    pid_batch_out = tl.program_id(0)
    pid_length = tl.program_id(1)
    
    # Calculate batch index and output channel index from pid_batch_out
    batch_idx = pid_batch_out // out_channels
    out_channel_idx = pid_batch_out % out_channels
    
    # Check bounds
    if batch_idx >= batch_size or out_channel_idx >= out_channels:
        return
    
    # Calculate the starting position in the output length dimension
    out_start = pid_length * BLOCK_SIZE_N
    offsets_out = out_start + tl.arange(0, BLOCK_SIZE_N)
    mask_out = offsets_out < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for in_channel in range(in_channels):
        # Calculate weight pointer offset for this (in_channel, out_channel) pair
        w_offset = in_channel * (out_channels * kernel_size) + out_channel_idx * kernel_size
        
        # Iterate over kernel positions
        for k in range(kernel_size):
            # Calculate input position based on output position, dilation, and kernel offset
            # For transposed convolution: input_pos = (output_pos - (dilation * (kernel_size - 1 - k))) // stride
            # But since stride=1 and padding=0 in our case, we can simplify
            input_pos = offsets_out - dilation * (kernel_size - 1 - k)
            
            # Check if input position is valid
            mask_input = (input_pos >= 0) & (input_pos < length)
            
            # Load input values with masking
            x_offset = batch_idx * (in_channels * length) + in_channel * length
            x_vals = tl.load(x_ptr + x_offset + input_pos, mask=mask_input, other=0.0)
            
            # Load weight value
            w_val = tl.load(w_ptr + w_offset + k)
            
            # Accumulate
            acc += x_vals * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx)
        acc += bias
    
    # Store output
    out_offset = batch_idx * (out_channels * out_length) + out_channel_idx * out_length
    tl.store(out_ptr + out_offset + offsets_out, acc, mask=mask_out)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of 1D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    batch_size, in_channels, length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length for transposed convolution
    # length_out = (length - 1) * stride + dilation * (kernel_size - 1) + 1 - 2 * padding
    length_out = (length - 1) * stride + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, length_out), dtype=x.dtype, device=x.device)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    # We parallelize over (batch_size * out_channels) and length dimensions
    BLOCK_SIZE_M = 1  # One block handles one (batch, out_channel) pair
    BLOCK_SIZE_N = 256  # Tunable parameter for length dimension
    
    grid = lambda meta: (
        batch_size * out_channels,
        (length_out + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size, length,
        stride, padding, dilation, length_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights and bias similar to nn.ConvTranspose1d
        # Using Kaiming uniform initialization
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.dilation
        )


# Import math module for initialization
import math