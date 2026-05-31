import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    norm_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute the sum of squares for Frobenius norm calculation.
    Uses a two-pass approach with atomic adds for partial sums.
    """
    # Compute partial sums in parallel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute square
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    squares = x * x
    
    # Accumulate to global sum using atomic add
    tl.atomic_add(norm_ptr, squares.sum())


@triton.jit
def divide_by_norm_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Divide input by precomputed Frobenius norm.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm and divide
    norm = tl.load(norm_ptr)
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor):
    """
    Optimized Frobenius norm normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # First, compute sum of squares
    # Use a small tensor for the norm accumulation (just 1 element)
    norm_sum = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    # Launch kernel to compute sum of squares
    frobenius_norm_kernel[grid_size](x, norm_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute the actual Frobenius norm (sqrt of sum of squares)
    norm = torch.sqrt(norm_sum)
    
    # Launch kernel to divide by norm
    divide_by_norm_kernel[grid_size](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using optimized Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return triton_frobenius_norm(x)