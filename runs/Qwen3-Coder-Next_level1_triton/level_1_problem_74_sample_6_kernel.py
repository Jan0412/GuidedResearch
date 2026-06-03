import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, length)
    w_ptr,  # Weight tensor (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor (out_channels,) or None
    out_ptr,  # Output tensor (batch, out_channels, out_length)
    batch_size, 
    in_channels, 
    out_channels, 
    in_length, 
    out_length, 
    kernel_size, 
    stride, 
    padding, 
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and output channel indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Output position
    out_pos = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_pos < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for in_channel in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate input position considering stride, padding, and dilation
            input_pos = (out_pos - (k * dilation)) // stride + padding
            
            # Check if input position is valid
            in_mask = (input_pos >= 0) & (input_pos < in_length) & ((out_pos - (k * dilation)) % stride == 0)
            valid_mask = out_mask & in_mask
            
            # Load input values
            x_offset = batch_idx * in_channels * in_length + in_channel * in_length + input_pos
            x_val = tl.load(x_ptr + x_offset, mask=valid_mask, other=0.0)
            
            # Load weight values
            w_offset = in_channel * out_channels * kernel_size + out_channel_idx * kernel_size + k
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        b_offset = out_channel_idx
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = batch_idx * out_channels * out_length + out_channel_idx * out_length + out_pos
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, in_length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Configure grid
    BLOCK_SIZE = 256
    grid = lambda meta: (
        batch_size,
        out_channels,
        triton.cdiv(out_length, meta["BLOCK_SIZE"]),
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_length, out_length,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 1D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )