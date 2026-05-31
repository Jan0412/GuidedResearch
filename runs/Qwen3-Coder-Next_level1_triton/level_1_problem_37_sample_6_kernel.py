import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,  # Input tensor pointer
    norm_ptr,  # Output norm pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to compute sum of squares for Frobenius norm."""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_squares = tl.sum(x * x, axis=0)
    
    # Use atomic add to accumulate results from all blocks
    tl.atomic_add(norm_ptr, sum_squares)


@triton.jit
def normalize_kernel(
    x_ptr,  # Input tensor pointer
    norm_ptr,  # Precomputed norm value
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to divide tensor by Frobenius norm."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm value (scalar)
    norm = tl.load(norm_ptr)
    
    # Normalize and store
    out = x / norm
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Applies Frobenius norm normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Create output tensor for norm (scalar)
    norm_tensor = torch.zeros(1, device=x.device, dtype=torch.float32)
    
    # First kernel: compute sum of squares
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    frobenius_norm_kernel[grid](x, norm_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute actual Frobenius norm (square root of sum of squares)
    norm = torch.sqrt(norm_tensor[0])
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Second kernel: normalize the tensor
    normalize_kernel[grid](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_normalize(x)