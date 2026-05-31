import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_reduction_kernel(
    x_ptr,
    sum_sq_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute the sum of squares of the input tensor elements.
    Uses atomic addition to accumulate partial sums from different blocks.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares for this block
    sum_sq = tl.sum(x * x, mask=mask, other=0.0)
    
    # Atomically add the block's sum to the global accumulator
    tl.atomic_add(sum_sq_ptr, sum_sq, mask=mask)


@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    norm,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to divide the input tensor by the Frobenius norm.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Frobenius norm normalization using custom Triton kernels.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Normalized tensor.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Initialize global sum of squares accumulator
    sum_sq = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    # Grid size for the reduction kernel
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    
    # Launch reduction kernel
    frobenius_norm_reduction_kernel[grid](
        x_ptr=x,
        sum_sq_ptr=sum_sq,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute the Frobenius norm
    norm = torch.sqrt(sum_sq).item()
    
    # Avoid division by zero
    if norm == 0.0:
        return out
    
    # Grid size for the normalization kernel
    grid = lambda meta: (num_blocks,)
    
    # Launch normalization kernel
    normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        norm=norm,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)