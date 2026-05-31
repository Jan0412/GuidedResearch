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
    Compute the sum of squares for Frobenius norm.
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    
    # Use atomic add to accumulate across blocks
    tl.atomic_add(norm_ptr, sum_sq)


@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize the input by the precomputed norm.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm (scalar)
    norm = tl.load(norm_ptr)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Apply Frobenius norm normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    # Prepare output tensor for sum of squares (scalar)
    sum_sq = torch.zeros(1, device=x.device, dtype=x.dtype)
    
    # Compute sum of squares
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    frobenius_norm_kernel[grid](x, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute final norm (sqrt of sum of squares)
    norm = torch.sqrt(sum_sq)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Normalize
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