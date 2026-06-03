import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,              # Input tensor: (batch_size, in_channels, length)
    w_ptr,              # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,              # Bias tensor: (out_channels,) or None
    out_ptr,            # Output tensor: (batch_size, out_channels, out_length)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    out_channels,       # Number of output channels
    length,             # Input length
    out_length,         # Output length
    kernel_size,        # Kernel size
    stride,             # Stride
    padding,            # Padding
    dilation,           # Dilation
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_out_pos = tl.program_id(2)
    
    # Calculate output position and handle boundary conditions
    out_pos = pid_out_pos
    
    # Skip if out of bounds
    if out_pos >= out_length:
        return
    
    # Calculate input position that contributes to this output position
    # For transposed convolution: out_pos = i * stride + j * dilation - padding
    # where i is input position and j is kernel position
    # Rearranging: i = (out_pos + padding - j * dilation) / stride
    
    # Accumulator for this output position
    acc = 0.0
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel positions
        for j in range(kernel_size):
            # Calculate corresponding input position
            i = out_pos + padding - j * dilation
            # Check if input position is valid and divisible by stride
            if i >= 0 and i % stride == 0:
                i = i // stride
                if i < length:
                    # Load input value
                    x_val = tl.load(x_ptr + pid_batch * in_channels * length + ic * length + i)
                    # Load weight value
                    w_idx = ic * out_channels * kernel_size + pid_out_channel * kernel_size + j
                    w_val = tl.load(w_ptr + w_idx)
                    # Multiply and accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_out_channel)
        acc += b_val
    
    # Store result
    out_ptr[pid_batch * out_channels * out_length + pid_out_channel * out_length + out_pos] = acc


# Optimized version using tiling for better performance
@triton.jit
def conv1d_transpose_kernel_tiled(
    x_ptr,              # Input tensor: (batch_size, in_channels, length)
    w_ptr,              # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,              # Bias tensor: (out_channels,) or None
    out_ptr,            # Output tensor: (batch_size, out_channels, out_length)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    out_channels,       # Number of output channels
    length,             # Input length
    out_length,         # Output length
    kernel_size,        # Kernel size
    stride,             # Stride
    padding,            # Padding
    dilation,           # Dilation
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
    BLOCK_SIZE_L: tl.constexpr,  # Block size for input positions
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    
    # Calculate output position range for this block
    out_pos_start = tl.program_id(2) * BLOCK_SIZE_L
    if out_pos_start >= out_length:
        return
    
    # Create output position offsets
    out_pos_offsets = tl.arange(0, BLOCK_SIZE_L)
    out_pos = out_pos_start + out_pos_offsets
    mask_out = out_pos < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel positions
        for j in range(kernel_size):
            # Calculate corresponding input position for each output position
            i = out_pos + padding - j * dilation
            # Check if input position is valid and divisible by stride
            valid_mask = (i >= 0) & (i % stride == 0) & (i < length)
            i = i // stride
            
            # Load input values
            x_offsets = pid_batch * in_channels * length + ic * length + i
            x_val = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
            
            # Load weight value
            w_idx = ic * out_channels * kernel_size + pid_out_channel * kernel_size + j
            w_val = tl.load(w_ptr + w_idx)
            
            # Multiply and accumulate
            acc += tl.where(valid_mask, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_out_channel)
        acc += b_val
    
    # Store results
    out_offsets = pid_batch * out_channels * out_length + pid_out_channel * out_length + out_pos
    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


def triton_conv1d_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length for transposed convolution
    out_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Determine kernel parameters
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 4   # Batch elements per block
    BLOCK_SIZE_K = 32  # Kernel elements per block
    BLOCK_SIZE_L = 64  # Output positions per block
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size,
        out_channels,
        triton.cdiv(out_length, meta["BLOCK_SIZE_L"])
    )
    
    # Launch the kernel
    conv1d_transpose_kernel_tiled[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        length, out_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        # Using the same initialization as PyTorch
        k = 1.0 / (in_channels * kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, 
                                       self.stride, self.padding, self.dilation)


import math