import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, kernel_size,
    input_length, output_length, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one (batch, out_channel) pair
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Create a range for the output length
    out_offsets = tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < output_length
    
    # Initialize accumulator for this output position
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for in_channel_idx in range(in_channels):
        # Iterate over kernel positions
        for k in range(kernel_size):
            # Calculate the corresponding input position
            # For transposed convolution: out_pos = in_pos * stride + k - padding + output_padding_offset
            # More precisely: in_pos = (out_pos + padding - k - output_padding_offset) // stride
            # We need to find valid input positions that contribute to each output position
            
            # Compute input position for each output position
            # in_pos = (out_pos + padding - k) // stride
            # But we need to handle the output_padding_offset
            # output_pos = in_pos * stride + k - padding + output_padding
            # So in_pos = (out_pos + padding - k - output_padding) // stride
            
            # Calculate the offset in the input tensor
            # input_offset = batch_idx * (in_channels * input_length) + in_channel_idx * input_length + in_pos
            # We need to compute in_pos for each out_pos
            
            # Vectorized calculation: for each out_pos, find valid in_pos values
            # in_pos = (out_pos + padding - k) // stride, but only if (out_pos + padding - k) % stride == 0
            # and 0 <= in_pos < input_length
            
            out_pos = out_offsets
            numerator = out_pos + padding - k
            # Check if divisible by stride
            is_valid = (numerator % stride == 0)
            in_pos = numerator // stride
            
            # Additional validity check
            in_valid = is_valid & (in_pos >= 0) & (in_pos < input_length)
            in_valid = in_valid & out_mask
            
            # Load input values
            input_offset = batch_idx * (in_channels * input_length) + in_channel_idx * input_length + in_pos
            x_val = tl.load(x_ptr + input_offset, mask=in_valid, other=0.0)
            
            # Load weight value
            weight_offset = in_channel_idx * (out_channels * kernel_size) + out_channel_idx * kernel_size + k
            w_val = tl.load(w_ptr + weight_offset)
            
            # Accumulate
            acc = tl.where(in_valid, acc + x_val * w_val, acc)
    
    # Add bias if available
    if b_ptr is not None:
        bias_offset = out_channel_idx
        bias_val = tl.load(b_ptr + bias_offset)
        acc = acc + bias_val
    
    # Store result
    output_offset = batch_idx * (out_channels * output_length) + out_channel_idx * output_length + out_offsets
    tl.store(y_ptr + output_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv1d_transpose(x, weight, bias, stride, padding, output_padding):
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + output_padding + kernel_size
    output_length = (input_length - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Prepare output tensor
    y = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)
    
    # Grid: (batch_size, out_channels)
    grid = (batch_size, out_channels)
    
    # Block size - tune for performance
    BLOCK_SIZE = 128
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels, kernel_size,
        input_length, output_length, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with custom Triton kernel.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Note: groups != 1 not supported in this simplified implementation
        assert groups == 1, "Groups > 1 not supported in Triton kernel"
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call our Triton-based transposed convolution
        return triton_conv1d_transpose(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding
        )