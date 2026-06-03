import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels // groups, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length,
    kernel_size, stride, padding, dilation, groups,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel computation
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    
    # Offset for batch
    batch_offset = pid_batch * in_channels * length
    
    # Compute output channel offset
    out_channel_offset = pid_out_channel
    
    # Create ranges for output positions
    out_pos_start = 0
    for out_pos in range(out_length):
        # Calculate input position for this output position
        in_pos = out_pos * stride - padding
        
        # Accumulator for convolution result
        acc = 0.0
        
        # Iterate over input channels (grouped)
        for in_channel_group in range(in_channels // groups):
            # Calculate global input channel
            in_channel = in_channel_group + (pid_out_channel // (out_channels // groups)) * (in_channels // groups)
            
            # Iterate over kernel positions
            for k in range(kernel_size):
                input_pos = in_pos + k * dilation
                if 0 <= input_pos < length:
                    # Calculate pointers
                    x_idx = batch_offset + in_channel * length + input_pos
                    w_idx = out_channel_offset * (in_channels // groups) * kernel_size + \
                            in_channel_group * kernel_size + k
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val
        
        # Add bias if present
        if b_ptr is not None:
            acc += tl.load(b_ptr + out_channel_offset)
        
        # Store result
        out_idx = pid_batch * out_channels * out_length + out_channel_offset * out_length + out_pos
        tl.store(out_ptr + out_idx, acc)


def triton_conv1d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # We use a 2D grid: batch_size x out_channels
    grid = (batch_size, out_channels)
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length,
        kernel_size, stride, padding, dilation, groups,
        BLOCK_SIZE_M=1,  # No blocking on output channels dimension
        BLOCK_SIZE_N=1,  # No blocking on batch dimension
        BLOCK_SIZE_K=32,  # Block size for kernel computation
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call the Triton convolution kernel
        return triton_conv1d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)

# Import math for initialization
import math