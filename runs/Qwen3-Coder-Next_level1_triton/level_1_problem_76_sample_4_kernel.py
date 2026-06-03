import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, output_length)
    n_batch, n_in_channels, n_out_channels, length, output_length, kernel_size,
    stride, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel
    BLOCK_SIZE_L: tl.constexpr,  # Block size for output length
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Batch block
    pid_l = tl.program_id(2)  # Output length block
    
    # Create output channel offsets
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = off_m < n_out_channels
    
    # Create batch offsets
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = off_n < n_batch
    
    # Create output length offsets
    off_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    l_mask = off_l < output_length
    
    # Output length * stride + (kernel_size-1)*dilation should be < length
    # We need to compute input positions for each output position
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over input channels
    for off_k in range(0, n_in_channels, BLOCK_SIZE_K):
        # Get input channel indices
        off_k_local = off_k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = off_k_local < n_in_channels
        k_mask_3d = k_mask[:, None, None]  # Shape (BLOCK_SIZE_K, 1, 1)
        
        # Load input: shape (batch, in_channels, length)
        # We need to access: y[batch, out_channel, out_pos] += x[batch, in_channel, input_pos] * w[out_channel, in_channel, kernel_pos]
        # For each output position l, input position = l * stride + kernel_pos * dilation
        
        # Process each kernel position
        for kernel_offset in range(kernel_size):
            # Compute input positions for this kernel offset
            input_pos = off_l * stride + kernel_offset * dilation
            input_pos_mask = input_pos < length
            
            # Create mask for valid input positions
            combined_mask = input_pos_mask[None, None, :] & n_mask[None, :, None] & m_mask[:, None, None]
            
            # Load input values: x[batch, in_channel, input_pos]
            # We need to gather from different positions, so we'll use a loop over input channels
            for ik in range(BLOCK_SIZE_K):
                ik_idx = off_k + ik
                if ik_idx < n_in_channels:
                    # Get indices for this input channel
                    x_indices = off_n[None, :] * (n_in_channels * length) + ik_idx * length + input_pos[None, :]
                    x_indices = x_indices.flatten()
                    x_values = tl.load(x_ptr + x_indices, mask=combined_mask.flatten(), other=0.0)
                    x_values = x_values.reshape(BLOCK_SIZE_N, BLOCK_SIZE_L)
                    
                    # Load weight values: w[out_channel, in_channel, kernel_offset]
                    w_indices = off_m[:, None] * (n_in_channels * kernel_size) + ik_idx * kernel_size + kernel_offset
                    w_values = tl.load(w_ptr + w_indices, mask=m_mask[:, None], other=0.0)
                    
                    # Compute outer product and accumulate
                    acc += w_values[:, :, None] * x_values[None, :, :]
    
    # Add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + off_m, mask=m_mask)
        acc += b[:, None, None]
    
    # Store output
    y_indices = off_m[:, None, None] * (n_batch * output_length) + off_n[None, :, None] * output_length + off_l[None, None, :]
    y_indices = y_indices.flatten()
    acc_flat = acc.flatten()
    tl.store(y_ptr + y_indices, acc_flat, mask=combined_mask.flatten())


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: int = 1, dilation: int = 1):
    """
    Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of convolution
        dilation: Dilation factor
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_length)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Tunable parameters for block sizes
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE_N = 8   # Batch size per block
    BLOCK_SIZE_K = 8   # Input channels per block
    BLOCK_SIZE_L = 64  # Output length per block
    
    # Grid dimensions
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(batch_size, BLOCK_SIZE_N),
        triton.cdiv(output_length, BLOCK_SIZE_L)
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels, length, output_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.dilation = dilation
        
        # Initialize weights
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
    
    def extra_repr(self):
        return 'in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, dilation={dilation}, bias={bias is not None}'.format(**self.__dict__)


# Import math for initialization
import math