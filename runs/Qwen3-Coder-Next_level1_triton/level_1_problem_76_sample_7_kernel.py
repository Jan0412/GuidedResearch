import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for 1D convolution with support for stride, dilation, and bias
@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length, kernel_size,
    stride: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_pos = tl.program_id(2)
    
    # Compute output position
    out_pos = pid_out_pos * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_out = out_pos < out_length
    
    # Calculate corresponding input positions for the first output position
    # Input position for output at out_pos: start = out_pos * stride - (kernel_size // 2) * dilation
    # But for exact convolution: start = out_pos * stride
    # Adjust for asymmetric padding: we assume valid convolution (no padding)
    start_in = out_pos * stride
    
    # Create offsets for kernel positions
    kernel_offsets = tl.arange(0, BLOCK_SIZE_K)
    mask_kernel = kernel_offsets < kernel_size
    
    # Create offsets for input channels
    in_c_offsets = tl.arange(0, BLOCK_SIZE_K)
    mask_in_c = in_c_offsets < in_channels
    
    # Compute weight pointer offset for this output channel
    w_offset = pid_out_c * (in_channels * kernel_size)
    
    # Accumulator for output
    output_sum = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for ic in range(0, in_channels, BLOCK_SIZE_K):
        # Load input values: x[batch, ic, start_in + k*dilation]
        # We need to handle boundary conditions carefully
        in_pos = start_in[:, None] + (kernel_offsets[None, :] * dilation)
        mask_in = (in_pos >= 0) & (in_pos < length) & mask_out[:, None]
        
        # Load input block
        x_offset = pid_batch * (in_channels * length) + ic * length + in_pos
        x_block = tl.load(x_ptr + x_offset, mask=mask_in, other=0.0)
        
        # Load weight block for current output channel and input channel
        w_block = tl.load(w_ptr + w_offset + ic * kernel_size + kernel_offsets, 
                         mask=mask_kernel)
        
        # Compute contribution: sum over kernel positions and input channel
        # x_block shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # w_block shape: (BLOCK_SIZE_K,)
        # We want to sum over the kernel positions for each output position
        contrib = tl.sum(x_block * w_block[None, :], axis=1)
        output_sum += contrib.to(tl.float32)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        output_sum += bias
    
    # Store result
    out_offset = pid_batch * (out_channels * out_length) + pid_out_c * out_length + out_pos
    tl.store(out_ptr + out_offset, output_sum.to(tl.float32), mask=mask_out)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Triton-based 1D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        dilation: Dilation rate
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, out_length), dtype=x.dtype, device=x.device)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    # We use 3D grid: (batch_size, out_channels, out_length // BLOCK_SIZE_N)
    BLOCK_SIZE_M = 1  # batch dimension
    BLOCK_SIZE_N = 32  # output position dimension
    BLOCK_SIZE_K = 32  # input channel dimension
    
    # Round up for grid calculation
    grid = (
        batch_size,
        out_channels,
        (out_length + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length, kernel_size,
        stride=stride, dilation=dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 1D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the convolution layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create the weight and bias parameters (same as nn.Conv1d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize parameters similar to nn.Conv1d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, dilation=self.dilation)