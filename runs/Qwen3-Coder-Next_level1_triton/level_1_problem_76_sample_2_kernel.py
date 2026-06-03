import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    y_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, 
    in_channels,
    out_channels,
    in_length,
    out_length,
    kernel_size,
    stride,
    dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel
):
    # Get program IDs
    pid_batch = tl.program_id(1)
    pid_out_channel = tl.program_id(0)
    
    # Offset for batch
    batch_offset = pid_batch * in_channels * in_length
    
    # Offset for output channel
    out_channel_offset = pid_out_channel * kernel_size * in_channels
    
    # Compute output position
    out_idx = tl.program_id(2)
    if out_idx >= out_length:
        return
    
    # Compute input position for this output
    in_idx = out_idx * stride
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Compute input pointer offset for this input channel
        x_offset = batch_offset + ic * in_length
        
        # Compute weight pointer offset for this input channel and output channel
        w_offset = out_channel_offset + ic * kernel_size
        
        # Load kernel weights
        k_offsets = tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < kernel_size
        kernel = tl.load(w_ptr + w_offset + k_offsets, mask=k_mask, other=0.0)
        
        # Load input values
        # Calculate positions for each kernel element
        positions = in_idx + k_offsets * dilation
        input_mask = positions < in_length
        
        # Load input values
        input_vals = tl.load(x_ptr + x_offset + positions, mask=input_mask, other=0.0)
        
        # Compute dot product for this input channel
        # Broadcast for vectorized computation
        acc += tl.sum(kernel * input_vals[None, :], axis=1)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_channel)
        acc += bias
    
    # Store result
    y_offset = pid_batch * out_channels * out_length + pid_out_channel * out_length + out_idx
    tl.store(y_ptr + y_offset, acc)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, dilation: int = 1):
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
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (in_length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Grid configuration: (out_channels, batch_size, out_length // BLOCK_SIZE_M)
    # We'll use a 3D grid where:
    # - pid0: output channel blocks
    # - pid1: batch blocks
    # - pid2: output position blocks
    
    # Set block sizes
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 8   # Batch per block
    BLOCK_SIZE_K = 16  # Kernel size per block
    
    # Compute grid dimensions
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (out_length + 1)  # One program per output position
    
    # Launch kernel
    conv1d_kernel[grid_m, grid_n, grid_k](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_length, out_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution model using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register the parameters but we'll implement our own forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights with the same pattern as nn.Conv1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            stride=self.stride, dilation=self.dilation)