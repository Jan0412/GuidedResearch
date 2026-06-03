import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,              # Input tensor (batch, in_channels, length)
    w_ptr,              # Weight tensor (out_channels, in_channels, kernel_size)
    b_ptr,              # Bias tensor (out_channels,) or None
    out_ptr,            # Output tensor (batch, out_channels, out_length)
    batch_size, 
    in_channels, 
    out_channels, 
    in_length, 
    out_length,
    kernel_size, 
    stride, 
    dilation,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_BLOCK: tl.constexpr,
):
    # Program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_pos = tl.program_id(2)
    
    # Calculate the input position this thread is responsible for
    in_pos = out_pos * stride
    
    # Compute output value
    acc = 0.0
    if b_ptr is not None:
        acc = tl.load(b_ptr + out_channel_idx)
    
    # Iterate over in_channels and kernel positions
    for in_c in range(in_channels):
        for k in range(kernel_size):
            # Calculate the actual input position with dilation
            input_idx = in_pos + k * dilation
            # Check bounds
            if input_idx >= 0 and input_idx < in_length:
                # Load input value
                x_offset = batch_idx * in_channels * in_length + in_c * in_length + input_idx
                x_val = tl.load(x_ptr + x_offset)
                
                # Load weight value
                w_offset = out_channel_idx * in_channels * kernel_size + in_c * kernel_size + k
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val
    
    # Store result
    out_offset = batch_idx * out_channels * out_length + out_channel_idx * out_length + out_pos
    tl.store(out_ptr + out_offset, acc)


def triton_conv1d(x, weight, bias, stride, dilation):
    """
    Triton-based 1D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Stride of convolution
        dilation: Dilation factor
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (in_length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Configure grid and block sizes
    # For large inputs, we use a 3D grid: [batch, out_channels, out_length_positions]
    # But for very large out_length, we'll use a different strategy
    
    # Use reasonable block sizes for good occupancy
    BLOCK_SIZE = 128
    KERNEL_BLOCK = 32  # Process kernel in blocks if needed
    
    # Grid dimensions: (batch_size, out_channels, out_length)
    grid = (batch_size, out_channels, out_length)
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_BLOCK=KERNEL_BLOCK,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 1D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights using Kaiming initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 
                                  (2.0 / (in_channels * kernel_size))**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out)
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)