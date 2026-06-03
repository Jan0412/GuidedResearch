import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, length_out)
    batch_size, in_channels, out_channels, kernel_size,
    input_length, output_length, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    block_start = tl.program_id(2) * BLOCK_SIZE
    
    # Compute output position range
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length
    
    # Compute the corresponding input positions for this output position
    # For transposed convolution: input_pos = (output_pos - output_padding - kernel_pos) / stride
    # We iterate over kernel positions
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel positions
    for k in range(kernel_size):
        # Compute input position for this kernel position
        # output_pos = input_pos * stride + kernel_pos - padding + output_padding
        # => input_pos = (output_pos - kernel_pos + padding - output_padding) / stride
        input_pos = (offsets - k + padding - output_padding) // stride
        
        # Check if input_pos is valid
        input_valid = (input_pos >= 0) & (input_pos < input_length) & ((offsets - k + padding - output_padding) % stride == 0)
        
        # Get input indices
        input_indices = input_pos * in_channels + tl.arange(0, 1)  # Will be broadcast
        
        # Get weight value for this kernel position and output channel
        w_offset = k * (out_channels * in_channels) + out_channel_idx * in_channels
        w_val = tl.load(w_ptr + w_offset + tl.arange(0, 1), mask=tl.arange(0, 1) < 1)  # Dummy mask
        
        # Loop over input channels
        for c in range(in_channels):
            # Load input value
            x_offset = batch_idx * (in_channels * input_length) + c * input_length + input_pos
            x_val = tl.load(x_ptr + x_offset, mask=input_valid, other=0.0)
            
            # Load weight value for this input channel
            w_idx = w_offset + c
            w_c = tl.load(w_ptr + w_idx, mask=tl.arange(0, 1) < 1)
            
            # Accumulate
            acc += tl.where(input_valid, x_val * w_c, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx)
        acc += bias
    
    # Store result
    out_offset = batch_idx * (out_channels * output_length) + out_channel_idx * output_length + offsets
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, output_padding):
    batch_size, in_channels, input_length = x.shape
    _, out_channels, kernel_size = weight.shape
    output_length = (input_length - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    grid = (batch_size, out_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        input_length, output_length, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with optimized Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the original layer
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias
        
        # Use the optimized Triton kernel for the main computation
        return triton_conv_transpose1d(x, weight, bias, 
                                      self.conv1d_transpose.stride[0],
                                      self.conv1d_transpose.padding[0],
                                      self.conv1d_transpose.output_padding[0])