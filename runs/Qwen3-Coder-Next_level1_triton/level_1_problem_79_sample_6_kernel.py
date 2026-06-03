import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    y_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    # Program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Calculate output position
    out_start = tl.program_id(2) * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for in_channel in range(in_channels):
        # Calculate input positions for this output position
        # For transposed conv: out_pos = in_pos * stride + (kernel_pos - padding - (kernel_size-1)*dilation)
        # Rearranging: in_pos = (out_pos - (kernel_pos - padding - (kernel_size-1)*dilation)) / stride
        
        # Process kernel positions
        for kernel_pos in range(kernel_size):
            # Calculate the input position that contributes to this output position
            # out_pos = in_pos * stride + (kernel_pos - padding) * dilation
            # => in_pos = (out_pos - (kernel_pos - padding) * dilation) / stride
            
            # Adjust for dilation and padding
            adjusted_kernel_pos = (kernel_pos - padding) * dilation
            in_pos_float = (out_offsets.to(tl.float32) - adjusted_kernel_pos) / stride
            
            # Check if in_pos is valid (integer and within bounds)
            # We need in_pos to be an integer for the transposed convolution
            is_valid = (out_offsets - adjusted_kernel_pos) % stride == 0
            in_pos = ((out_offsets - adjusted_kernel_pos) // stride).to(tl.int32)
            in_valid = (in_pos >= 0) & (in_pos < length) & is_valid
            
            # Load input value if valid
            x_offset = batch_id * (in_channels * length) + in_channel * length + in_pos
            x_val = tl.load(x_ptr + x_offset, mask=in_valid, other=0.0)
            
            # Load weight value
            w_offset = (in_channel * out_channels + out_channel_id) * kernel_size + kernel_pos
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val * in_valid.to(tl.float32)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_id)
        acc += bias
    
    # Store result
    y_offset = batch_id * (out_channels * out_length) + out_channel_id * out_length + out_offsets
    tl.store(y_ptr + y_offset, acc, mask=out_mask)


def triton_conv1d_transpose(x, weight, bias, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 1D transposed convolution.
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length for transposed convolution
    # out_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
    # Assuming out_padding=0 as not specified
    out_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Grid dimensions: (batch_size, out_channels, (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    BLOCK_SIZE = 128
    grid = (batch_size, out_channels, (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels, length, out_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights similar to nn.ConvTranspose1d
        # Using kaiming_uniform initialization like PyTorch
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        gain = nn.init.calculate_gain('leaky_relu', 0)  # default for ConvTranspose
        std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size).normal_(0, std))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, std))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 1D convolution using Triton.
        """
        # Ensure x is contiguous
        x = x.contiguous()
        
        # Call our Triton implementation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias if self.bias_flag else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )