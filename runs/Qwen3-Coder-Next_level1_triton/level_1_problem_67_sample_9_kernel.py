import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, output_length)
    batch_size, in_channels, out_channels, 
    input_length, output_length, kernel_size,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Batch size block
    BLOCK_SIZE_N: tl.constexpr,  # Output channels block
    BLOCK_SIZE_K: tl.constexpr,  # Input channels block
    BLOCK_SIZE_L: tl.constexpr,  # Output length block
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_out_len = tl.program_id(2)
    
    # Offset calculations
    out_ch_start = pid_out_ch * BLOCK_SIZE_N
    out_len_start = pid_out_len * BLOCK_SIZE_L
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Convolution loop over input channels and kernel positions
    for c in range(in_channels):
        for k in range(kernel_size):
            # Calculate input position with dilation
            input_pos = out_len_start * stride + k * dilation - padding
            
            # Check bounds for input position
            if input_pos >= 0 and input_pos < input_length:
                # Load input values: (BLOCK_SIZE_N, BLOCK_SIZE_L)
                x_offsets = (pid_batch * in_channels * input_length + 
                            c * input_length + input_pos)
                x_val = tl.load(x_ptr + x_offsets)
                
                # Load weight values
                w_offsets = (out_ch_start * in_channels * kernel_size + 
                            c * kernel_size + k)
                w_val = tl.load(w_ptr + w_offsets)
                
                # Accumulate: broadcast and multiply
                acc += x_val * w_val
    
    # Apply bias if present
    if b_ptr is not None:
        bias_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_N)
        bias_mask = bias_offsets < out_channels
        bias_val = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_val[:, None]
    
    # Store output
    out_offsets = (pid_batch * out_channels * output_length + 
                   (out_ch_start + tl.arange(0, BLOCK_SIZE_N)) * output_length + 
                   out_len_start + tl.arange(0, BLOCK_SIZE_L))
    out_mask = ((out_ch_start + tl.arange(0, BLOCK_SIZE_N))[:, None] < out_channels) & \
               (out_len_start + tl.arange(0, BLOCK_SIZE_L) < output_length)
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Performs 1D convolution using Triton kernel.
    Assumes groups == 1 (standard convolution)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Check for bias
    bias_ptr = bias.contiguous() if bias is not None else None
    
    # Configure kernel launch parameters
    # Block sizes tuned for GPU efficiency
    BLOCK_SIZE_M = 1    # Batch size block
    BLOCK_SIZE_N = 16   # Output channels block
    BLOCK_SIZE_K = 16   # Input channels block
    BLOCK_SIZE_L = 64   # Output length block
    
    # Grid dimensions
    grid = (
        batch_size,  # batch_size
        triton.cdiv(out_channels, BLOCK_SIZE_N),  # out_channels blocks
        triton.cdiv(output_length, BLOCK_SIZE_L),  # output_length blocks
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias_ptr, out,
        batch_size, in_channels, out_channels,
        input_length, output_length, kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights using Xavier/Glorot initialization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier initialization for convolutional layers
        fan_in = self.in_channels * self.kernel_size
        fan_out = self.out_channels * self.kernel_size
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        with torch.no_grad():
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)