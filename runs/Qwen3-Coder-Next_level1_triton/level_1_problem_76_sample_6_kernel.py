import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,              # Input tensor (batch_size, in_channels, length)
    w_ptr,              # Weight tensor (out_channels, in_channels, kernel_size)
    b_ptr,              # Bias tensor (out_channels,) or None
    out_ptr,            # Output tensor (batch_size, out_channels, out_length)
    batch_size, 
    in_channels,
    out_channels,
    length,             # Input length
    out_length,         # Output length
    kernel_size,
    stride,
    dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output length
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels * kernel_size
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Offset for batch processing
    batch_offset = pid_batch * in_channels * length
    
    # Calculate output channel range
    out_channel_start = pid_m * BLOCK_SIZE_M
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
    out_channel_mask = out_channel_offsets < out_channels
    
    # Calculate output length range
    out_length_start = pid_n * BLOCK_SIZE_N
    out_length_offsets = out_length_start + tl.arange(0, BLOCK_SIZE_N)
    out_length_mask = out_length_offsets < out_length
    
    # Create meshgrid for output channel and length
    out_channel_grid = out_channel_offsets[:, None]
    out_length_grid = out_length_offsets[None, :]
    
    # Accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution computation
    # For each position in output, compute dot product of input window and kernel
    for k in range(0, in_channels * kernel_size, BLOCK_SIZE_K):
        kernel_col = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Reshape to separate in_channels and kernel_size
        in_ch = kernel_col // kernel_size
        kernel_pos = kernel_col % kernel_size
        
        # Compute input positions for this convolution
        # out_length_idx * stride + kernel_pos * dilation
        input_positions = out_length_grid[None, :, :] * stride + kernel_pos[None, None, :] * dilation
        
        # Broadcast for batch and channel
        input_positions = input_positions + batch_offset + in_ch[:, None, None] * length
        
        # Flatten for loading
        input_positions_flat = tl.reshape(input_positions, (BLOCK_SIZE_M, BLOCK_SIZE_K // (in_channels * kernel_size) * in_channels * kernel_size, BLOCK_SIZE_N))
        
        # We need to compute input indices properly
        # Let's restructure to compute for each output element
        pass  # Placeholder for now, we'll use a different approach below
    
    # Simpler direct implementation for convolution
    # For each output position (n, c_out)
    # acc[c_out, n] = sum_{c_in, k} x[n, c_in, pos] * w[c_out, c_in, k]
    # where pos = n * stride + k * dilation
    
    # We'll use a nested loop approach for clarity and correctness
    for c_in in range(in_channels):
        for k in range(kernel_size):
            # Compute input positions for all output positions
            input_pos = out_length_offsets * stride + k * dilation
            input_pos = input_pos + batch_offset + c_in * length
            
            # Load input values: [BLOCK_SIZE_N]
            input_mask = (input_pos >= 0) & (input_pos < batch_offset + in_channels * length)
            x_val = tl.load(x_ptr + input_pos, mask=input_mask, other=0.0)
            
            # Load corresponding weights: [BLOCK_SIZE_M]
            weight_idx = (out_channel_grid * in_channels * kernel_size + 
                         c_in * kernel_size + k)
            w_val = tl.load(w_ptr + weight_idx, mask=out_channel_mask)
            
            # Broadcast and multiply: [BLOCK_SIZE_M, BLOCK_SIZE_N]
            x_broadcast = x_val[None, :]  # [1, BLOCK_SIZE_N]
            w_broadcast = w_val[:, None]  # [BLOCK_SIZE_M, 1]
            
            acc += w_broadcast * x_broadcast
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_grid, mask=out_channel_mask)
        acc += bias[:, None]
    
    # Convert to output type and store
    out_idx = (pid_batch * out_channels * out_length + 
               out_channel_grid * out_length + out_length_grid)
    out_mask = out_channel_mask[:, None] & out_length_mask[None, :]
    
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv1d(x, weight, bias=None, stride=1, dilation=1):
    """
    Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution
        dilation: Dilation for convolution
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    # Grid: (num_blocks_out_channels, num_blocks_out_length, batch_size)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32  # Not used in simplified version but kept for consistency
    
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(out_length, BLOCK_SIZE_N),
        batch_size,
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        length, out_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights with same shape as nn.Conv1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)