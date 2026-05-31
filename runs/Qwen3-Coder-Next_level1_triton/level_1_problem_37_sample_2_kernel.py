import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    n_elements,
    norm_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes the Frobenius norm of a tensor using a parallel reduction.
    """
    # Create offsets for the current block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute squared values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_squared = x * x
    
    # Perform reduction using tl.sum
    sum_squared = tl.sum(x_squared, axis=0)
    
    # Store partial sums
    tl.store(norm_ptr + pid, sum_squared)


@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalizes the input tensor by dividing by the precomputed Frobenius norm.
    """
    # Create offsets for the current block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load the Frobenius norm (scalar)
    norm = tl.load(norm_ptr)
    
    # Avoid division by zero
    norm_safe = tl.where(norm > 0, norm, 1.0)
    
    # Normalize and store
    out = x / norm_safe
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor):
    """
    Computes Frobenius norm normalization using Triton kernels.
    
    This implementation:
    1. Computes the sum of squares using a parallel reduction kernel
    2. Takes the square root to get the Frobenius norm
    3. Normalizes the input tensor in a separate kernel
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    
    # Determine optimal block size and grid size for reduction
    BLOCK_SIZE = 1024
    # Calculate number of blocks needed for reduction
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Handle edge case where n_elements is small
    if n_blocks == 0:
        n_blocks = 1
    
    # Allocate buffer for partial sums in reduction
    partial_sums = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    
    # Launch reduction kernel
    grid = lambda meta: (n_blocks,)
    frobenius_norm_kernel[grid](x, n_elements, partial_sums, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute total sum of squares by summing the partial sums on GPU
    # For very large tensors, we might need multiple reduction steps, but for simplicity
    # we'll do it in one step since n_blocks is typically small
    total_sum = torch.sum(partial_sums)
    
    # Compute Frobenius norm
    norm = torch.sqrt(total_sum)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Launch normalization kernel
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    normalize_kernel[grid](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
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