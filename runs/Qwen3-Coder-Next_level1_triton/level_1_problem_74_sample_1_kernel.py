import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, L_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch, out_channels, L_out)
    batch_size, in_channels, out_channels, 
    input_length, kernel_size, 
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, out_channel, output_position)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Global output index
    out_idx = pid_l
    
    # Calculate the output length for this dimension
    # L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # But we'll compute based on input and parameters
    
    # Check bounds
    if out_idx >= ((input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1):
        return
    
    # Calculate input position that contributes to this output position
    # For transposed convolution: out[i] = sum_j x[j] * w[j, i, k] where j = (i - k) // stride + something
    
    # The relationship: for output position l_out, it gets contributions from input positions l_in where
    # l_in * stride + k * dilation - padding = l_out
    # => l_in = (l_out + padding - k * dilation) / stride
    
    acc = 0.0
    
    # Loop over input channels and kernel positions
    for ic in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Compute corresponding input position
            l_in = (out_idx + padding - k * dilation) // stride
            
            # Check if this input position is valid and aligns with stride
            if (out_idx + padding - k * dilation) % stride == 0 and 0 <= l_in < input_length:
                # Calculate pointers
                x_offset = pid_b * (in_channels * input_length) + ic * input_length + l_in
                w_offset = ic * (out_channels * kernel_size) + pid_c * kernel_size + k
                
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_c
        acc += tl.load(b_ptr + bias_offset)
    
    # Store result
    out_offset = pid_b * (out_channels * ((input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)) + \
                 pid_c * ((input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1) + out_idx
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose1d(x, weight, bias, stride=1, padding=0, dilation=1):
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, L_in)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation: convolution parameters
    
    Returns:
        Output tensor of shape (batch, out_channels, L_out)
    """
    batch_size, in_channels, input_length = x.shape
    in_channels_w, out_channels, kernel_size = weight.shape
    
    assert in_channels == in_channels_w, "Input channels must match"
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Grid dimensions: (batch, out_channels, output_length)
    grid = (batch_size, out_channels, output_length)
    
    # Launch kernel
    BLOCK_SIZE = 1  # Not used in this implementation but required for kernel signature
    
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        input_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with the same parameters but we'll implement the forward pass manually
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )