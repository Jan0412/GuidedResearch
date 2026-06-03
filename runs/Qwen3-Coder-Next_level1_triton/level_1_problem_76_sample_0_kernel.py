import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, 
    input_length, kernel_size, 
    stride, dilation, out_length,
    BLOCK_M: tl.constexpr,  # Block size for output channels
    BLOCK_N: tl.constexpr,  # Block size for output positions
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_pos = tl.program_id(2)
    
    # Output position range
    out_start = pid_out_pos * BLOCK_N
    out_offsets = out_start + tl.arange(0, BLOCK_N)
    out_mask = out_offsets < out_length
    
    # Output channel range
    c_out_start = pid_out_c * BLOCK_M
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_M)
    c_out_mask = c_out_offsets < out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Convolution: for each output position, accumulate over input channels and kernel
    for k in range(kernel_size):
        # Compute input position for this kernel element
        # input_pos = out_pos * stride + k * dilation
        input_pos = out_offsets * stride + k * dilation
        input_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # For each input channel
        for c_in in range(in_channels):
            # Load input: (batch, c_in, input_pos)
            x_offset = pid_batch * (in_channels * input_length) + c_in * input_length + input_pos
            x_val = tl.load(x_ptr + x_offset, mask=input_mask, other=0.0)
            
            # Load weight: (c_out, c_in, k)
            w_offset = c_out_offsets * (in_channels * kernel_size) + c_in * kernel_size + k
            w_val = tl.load(w_ptr + w_offset, mask=c_out_mask[:, None], other=0.0)
            
            # Accumulate: (BLOCK_M, 1) * (1, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += w_val * x_val[None, :]
    
    # Convert accumulator to output dtype and add bias if present
    acc = acc.to(y_ptr.dtype.element_ty)
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = c_out_offsets
        b_val = tl.load(b_ptr + b_offset, mask=c_out_mask, other=0.0)
        acc += b_val[:, None]
    
    # Store output
    y_offset = pid_batch * (out_channels * out_length) + c_out_offsets * out_length + out_offsets
    y_mask = c_out_mask[:, None] & out_mask[None, :]
    tl.store(y_ptr + y_offset, acc, mask=y_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Triton-based 1D convolution with support for stride and dilation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length: out_length = floor((input_length - dilation * (kernel_size - 1) - 1) / stride) + 1
    out_length = (input_length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty((batch_size, out_channels, out_length), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    # Block sizes for optimization (tuned for large sequences)
    BLOCK_M = 16  # Output channels per block
    BLOCK_N = 128  # Output positions per block
    
    # Grid: (batch_size, out_channels // BLOCK_M, out_length // BLOCK_N)
    grid = (
        batch_size,
        triton.cdiv(out_channels, BLOCK_M),
        triton.cdiv(out_length, BLOCK_N)
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias,
        y,
        batch_size, in_channels, out_channels,
        input_length, kernel_size,
        stride, dilation, out_length,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create parameter placeholders - these will be populated during forward
        # We need to create the actual parameters to match torch.nn.Module behavior
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters (same as nn.Conv1d initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, dilation=self.dilation)


import math

# Override ModelNew to fix import
class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters (same as nn.Conv1d initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, dilation=self.dilation)