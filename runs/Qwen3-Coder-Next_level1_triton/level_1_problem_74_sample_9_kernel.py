import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,              # Input tensor: (batch, in_channels, length)
    w_ptr,              # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,              # Bias tensor: (out_channels,) or None
    y_ptr,              # Output tensor: (batch, out_channels, out_length)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    out_channels,       # Number of output channels
    in_length,          # Input length
    out_length,         # Output length
    kernel_size,        # Kernel size
    stride,             # Stride
    padding,            # Padding
    dilation,           # Dilation
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    # Parallelization: each program handles a batch-channel-output position
    pid_b = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # output channel index
    pid_out_pos = tl.program_id(2)  # output position index
    
    if pid_b >= batch_size or pid_oc >= out_channels or pid_out_pos >= out_length:
        return
    
    # Calculate the output position
    out_pos = pid_out_pos
    
    # Compute the starting input position for this output position
    # For transposed convolution: out_pos = i * stride + (k-1)*dilation - padding
    # So i = (out_pos + padding - (k-1)*dilation) / stride
    # We need to find all valid (input_pos, kernel_pos) pairs that contribute to this output
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Compute corresponding input position
            input_pos = (out_pos + padding - k * dilation) // stride
            
            # Check if input_pos is within valid range and satisfies the stride condition
            if (out_pos + padding - k * dilation) % stride == 0 and \
               input_pos >= 0 and input_pos < in_length:
                
                # Get pointers for current input position and kernel position
                x_offset = pid_b * in_channels * in_length + ic * in_length + input_pos
                w_offset = ic * out_channels * kernel_size + pid_oc * kernel_size + k
                
                # Load values
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_oc
        acc += tl.load(b_ptr + b_offset)
    
    # Store result
    y_offset = pid_b * out_channels * out_length + pid_oc * out_length + out_pos
    tl.store(y_ptr + y_offset, acc)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Performs transposed 1D convolution using Triton kernel.
    """
    batch_size, in_channels, in_length = x.shape
    in_channels_w, out_channels, kernel_size = weight.shape
    
    # Calculate output length: out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # We parallelize over batch, output channels, and output positions
    grid = (batch_size, out_channels, out_length)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_length, out_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE=1,
        DTYPE=tl.float32 if x.dtype == torch.float32 else tl.float16
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    Uses optimized Triton kernel instead of PyTorch's native implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights using same initialization as PyTorch's ConvTranspose1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


import math