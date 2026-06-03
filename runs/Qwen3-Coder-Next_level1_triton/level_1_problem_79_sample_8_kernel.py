import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch_size, out_channels, length_out)
    batch_size, in_channels, out_channels, length, length_out, kernel_size,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch_size
    BLOCK_SIZE_N: tl.constexpr,  # Block size for out_channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels
    BLOCK_SIZE_L: tl.constexpr,  # Block size for kernel_size
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_n = tl.program_id(1)  # out_channels index
    pid_m = tl.program_id(2)  # output position index
    
    # Offset for batch
    batch_offset = pid_b * in_channels * length
    
    # Offset for output channel
    out_channel_offset = pid_n * length_out
    
    # Output position
    out_pos = pid_m * BLOCK_SIZE_M
    
    # Compute output position range
    out_offsets = out_pos + tl.arange(0, BLOCK_SIZE_M)
    out_mask = out_offsets < length_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over in_channels
    for k in range(in_channels):
        # Input position offsets for this in_channel
        in_offsets_full = out_offsets * stride - padding + tl.arange(0, BLOCK_SIZE_L)[None, :] * dilation
        in_offsets_full = in_offsets_full + k * length
        in_offsets_full = tl.reshape(in_offsets_full, (BLOCK_SIZE_M * BLOCK_SIZE_L,))
        
        # Kernel position offsets
        kernel_offsets = k * out_channels * kernel_size + pid_n * kernel_size + tl.arange(0, BLOCK_SIZE_L)
        
        # Compute mask for valid input positions
        in_pos_mask = (in_offsets_full >= 0) & (in_offsets_full < length)
        kernel_mask = tl.arange(0, BLOCK_SIZE_L) < kernel_size
        
        # Load input values
        x_val = tl.load(x_ptr + in_offsets_full, mask=in_pos_mask, other=0.0)
        x_val = tl.reshape(x_val, (BLOCK_SIZE_M, BLOCK_SIZE_L))
        
        # Load kernel values
        w_val = tl.load(w_ptr + kernel_offsets, mask=kernel_mask, other=0.0)
        w_val = tl.reshape(w_val, (1, BLOCK_SIZE_L))
        
        # Accumulate: out[batch, out_ch, out_pos] += sum_k x[batch, in_ch, in_pos] * w[in_ch, out_ch, k]
        # where in_pos = out_pos * stride - padding + k * dilation
        acc += tl.sum(x_val * w_val, axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_n)
        acc += bias
    
    # Store output
    out_offsets_final = batch_offset + out_channel_offset + out_offsets
    tl.store(out_ptr + out_offsets_final, acc, mask=out_mask)


def triton_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """
    Triton implementation of 1D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length: length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # For our case with output_padding=0: length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, length_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 64  # Output position block size
    BLOCK_SIZE_N = 32  # Output channel block size  
    BLOCK_SIZE_K = 16  # Input channel block size
    BLOCK_SIZE_L = 16  # Kernel position block size
    
    # Grid dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (out_channels + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        (length_out + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"]
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, length_out, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation. Uses optimized Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers instead of nn.Parameter to avoid automatic gradient computation issues
        # We'll handle gradients manually in training scenarios
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using kaiming_uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv_transpose1d(x, self.weight, self.bias, 
                                       self.stride, self.padding, self.dilation)