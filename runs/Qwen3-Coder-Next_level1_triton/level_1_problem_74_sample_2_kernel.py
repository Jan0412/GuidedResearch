import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, length)
    w_ptr,  # Weight tensor (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor (out_channels,) or None
    out_ptr,  # Output tensor (batch, out_channels, out_length)
    n_batch, n_in_channels, n_out_channels, kernel_size,
    in_length, out_length,
    stride, padding, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_L: tl.constexpr,  # Block size for length dimension
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Create output position ranges
    out_ch_offsets = out_ch_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ch_mask = out_ch_offsets < n_out_channels
    
    # Calculate output length blocks
    out_l_start = tl.program_id(2) * BLOCK_SIZE_L
    out_l_offsets = out_l_start + tl.arange(0, BLOCK_SIZE_L)
    out_l_mask = out_l_offsets < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for in_ch_id in range(0, n_in_channels, BLOCK_SIZE_K):
        in_ch_offsets = in_ch_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        in_ch_mask = in_ch_offsets < n_in_channels
        
        # Load input block: (BLOCK_SIZE_K, in_length)
        x_block = tl.load(
            x_ptr + batch_id * n_in_channels * in_length + in_ch_offsets[:, None] * in_length,
            mask=in_ch_mask[:, None],
            other=0.0
        )
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate corresponding input position for this kernel position
            # For transposed conv: out_pos = in_pos * stride + (k - padding) * dilation
            # So in_pos = (out_pos - (k - padding) * dilation) / stride
            # Only valid when (out_pos - (k - padding) * dilation) is divisible by stride
            
            # Calculate offset from output position to input position
            offset = (k - padding) * dilation
            
            # Input positions corresponding to output positions
            in_l_offsets = (out_l_offsets - offset) // stride
            
            # Check which output positions have valid input positions
            valid_mask = ((out_l_offsets - offset) % stride == 0) & \
                        (in_l_offsets >= 0) & (in_l_offsets < in_length)
            
            # Load kernel weights for this kernel position
            w_block = tl.load(
                w_ptr + in_ch_offsets[:, None] * n_out_channels * kernel_size + \
                       out_ch_offsets[None, :] * kernel_size + k,
                mask=in_ch_mask[:, None] & out_ch_mask[None, :],
                other=0.0
            )
            
            # Gather input values at valid positions
            in_vals = tl.where(
                valid_mask[None, :],
                tl.load(
                    x_block + in_l_offsets[None, :],
                    mask=valid_mask[None, :],
                    other=0.0
                ),
                0.0
            )
            
            # Accumulate: output[out_ch, out_l] += sum_over_in_ch(input[in_ch, in_l] * weight[in_ch, out_ch, k])
            # We need to compute outer product for accumulation
            acc += tl.dot(w_block, in_vals, out_dtype=tl.float32)
    
    # Convert to output type and apply bias if present
    out = acc.to(x_ptr.dtype.element_ty)
    
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_ch_offsets, mask=out_ch_mask)
        out = out + bias[:, None]
    
    # Store result
    tl.store(
        out_ptr + batch_id * n_out_channels * out_length + 
                out_ch_offsets[:, None] * out_length + out_l_offsets[None, :],
        out,
        mask=out_ch_mask[:, None] & out_l_mask[None, :]
    )


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose1d.
    
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
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_length = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    
    # Calculate output length for transposed convolution
    # out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
    # Since out_padding=0 in our case:
    out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    # Grid dimensions: (batch, out_channels_blocks, length_blocks)
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 16  # Tune for your GPU
    BLOCK_SIZE_K = 16  # Tune for your GPU  
    BLOCK_SIZE_L = 128  # Tune for your GPU
    
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["BLOCK_SIZE_N"]),
        triton.cdiv(out_length, meta["BLOCK_SIZE_L"])
    )
    
    # Launch the kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        in_length, out_length,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 1D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the convolution layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters (same as nn.ConvTranspose1d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights and bias similar to nn.ConvTranspose1d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None}')


# Import math for kaiming initialization
import math