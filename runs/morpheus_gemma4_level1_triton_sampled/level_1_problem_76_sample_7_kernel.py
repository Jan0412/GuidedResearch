import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, L, Cout, K, S, D, Lout,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Parallelize over (batch * out_channels) and (length_out)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Map pid_m to batch index and output channel index
    b_idx = pid_m // Cout
    co_idx = pid_m % Cout

    # Map pid_n to output length range
    lo_start = pid_n * BLOCK_N
    lo_offsets = lo_start + tl.arange(0, BLOCK_N)
    mask_lo = lo_offsets < Lout

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over input channels in blocks
    for ci_start in range(0, Cin, BLOCK_K):
        ci_offsets = ci_start + tl.arange(0, BLOCK_K)
        mask_ci = ci_offsets < Cin

        # Loop over the kernel size
        for k in range(K):
            # Calculate indices for input x: (B, Cin, L)
            # x[b_idx, ci_offsets, lo_offsets * S + k * D]
            # We need to load a block of (BLOCK_K, BLOCK_N)
            x_ptr_off = (
                b_idx * Cin * L 
                + ci_offsets[:, None] * L 
                + (lo_offsets[None, :] * S + k * D)
            )
            x_val = tl.load(x_ptr + x_ptr_off, mask=mask_ci[:, None] & mask_lo[None, :], other=0.0)

            # Calculate indices for weight w: (Cout, Cin, K)
            # w[co_idx, ci_offsets, k]
            w_ptr_off = co_idx * Cin * K + ci_offsets * K + k
            w_val = tl.load(w_ptr + w_ptr_off, mask=mask_ci, other=0.0)

            # Multiply and accumulate along the Cin dimension
            acc += tl.sum(x_val * w_val[:, None], axis=0)

    # Add bias
    bias_val = tl.load(b_ptr + co_idx) if b_ptr is not None else 0.0
    acc += bias_val

    # Store the result in out: (B, Cout, Lout)
    out_ptr_off = b_idx * Cout * Lout + co_idx * Lout + lo_offsets
    tl.store(out_ptr + out_ptr_off, acc, mask=mask_lo)


def triton_conv1d(x, weight, bias, stride, dilation):
    # x: (B, Cin, L)
    # weight: (Cout, Cin, K)
    # bias: (Cout,)
    B, Cin, L = x.shape
    Cout, _, K = weight.shape
    
    # Calculate Lout
    Lout = (L - dilation * (K - 1) - 1) // stride + 1
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, Cout, Lout), device=x.device, dtype=x.dtype)
    
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Grid: (B * Cout, ceil(Lout / BLOCK_N))
    grid = (B * Cout, (Lout + BLOCK_N - 1) // BLOCK_N)
    
    conv1d_kernel[grid](
        x, weight, bias, out,
        B, Cin, L, Cout, K, stride, dilation, Lout,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution implementation using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Use nn.Parameter to maintain weights and bias for compatibility with PyTorch optimizers
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.dilation
        )