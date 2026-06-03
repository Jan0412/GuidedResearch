import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor (B, C_in, L_in)
    w_ptr,  # Weight tensor (C_in, C_out, K)
    b_ptr,  # Bias tensor (C_out,) or nullptr
    out_ptr,  # Output tensor (B, C_out, L_out)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    L_in: tl.constexpr,
    L_out: tl.constexpr,
    K: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output length
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l_out = tl.program_id(2)
    
    # Calculate offsets for output
    out_offsets = pid_l_out * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_mask = out_offsets < L_out
    
    # Output pointer offset for this batch and output channel
    out_batch_offset = pid_batch * C_out * L_out + pid_c_out * L_out
    out_ptr_batch = out_ptr + out_batch_offset
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(C_in):
        # Calculate input position corresponding to each output position
        # For transposed convolution: l_in = (l_out - (K-1)*dilation + padding) // stride
        # But we need to check all possible input positions that contribute to each output
        
        # For each output position, we need to check all kernel positions
        kernel_offsets = tl.arange(0, BLOCK_SIZE_K) if BLOCK_SIZE_K == K else tl.arange(0, K)
        
        # For this output position and kernel position, calculate input position
        l_out_range = pid_l_out * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        l_out_range = tl.reshape(l_out_range, (BLOCK_SIZE_N, 1))
        kernel_range = tl.reshape(tl.arange(0, K), (1, K))
        
        # Calculate input position: l_in = (l_out - (k-1)*dilation - padding) / stride
        l_in_range = (l_out_range - (kernel_range - 1) * dilation - padding) // stride
        
        # Check if input position is valid
        valid_mask = (l_in_range >= 0) & (l_in_range < L_in)
        
        # Load input values
        x_batch_offset = pid_batch * C_in * L_in + c_in * L_in
        x_ptr_batch = x_ptr + x_batch_offset
        
        # Load input at valid positions
        x_offsets = l_in_range.flatten()
        x_mask = (x_offsets >= 0) & (x_offsets < L_in)
        
        # For valid positions, load input
        x_val = tl.load(x_ptr_batch + x_offsets, mask=x_mask, other=0.0)
        x_val = tl.reshape(x_val, (BLOCK_SIZE_N, K))
        
        # Load weights: w[c_in, pid_c_out, k]
        w_ptr_offset = c_in * C_out * K + pid_c_out * K
        w_val = tl.load(w_ptr + w_ptr_offset + kernel_range.flatten(), mask=tl.reshape(kernel_range.flatten() < K, (K,)))
        w_val = tl.reshape(w_val, (1, K))
        
        # Multiply and accumulate for valid positions
        contrib = x_val * w_val * tl.reshape(valid_mask, (BLOCK_SIZE_N, K)).to(tl.float32)
        acc += tl.sum(contrib, axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    tl.store(out_ptr_batch + out_offsets, acc, mask=out_mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Performs transposed 1D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, length_out)
    """
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, L_out, dtype=x.dtype, device=x.device)
    
    # Configure grid and block sizes
    # Grid: (batch_size, out_channels, output_length_blocks)
    BLOCK_SIZE_M = 16  # For output channels
    BLOCK_SIZE_N = 256  # For output length
    BLOCK_SIZE_K = K  # For input channels (use full kernel size)
    
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, BLOCK_SIZE_M),
        triton.cdiv(L_out, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B=B, C_in=C_in, C_out=C_out, L_in=L_in, L_out=L_out, K=K,
        stride=stride, padding=padding, dilation=dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    Uses optimized Triton kernel instead of PyTorch's native implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers for weight and bias to match nn.ConvTranspose1d behavior
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights (matching nn.ConvTranspose1d initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


# Import math for initialization
import math