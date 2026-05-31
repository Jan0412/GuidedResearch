import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    n_elements,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values for accumulation
    x_sq = x * x
    
    # Use tl.sum for the block reduction
    acc = tl.sum(x_sq)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), acc)


@triton.jit
def frobenius_norm_final_kernel(
    partial_sums_ptr,
    n_partial_sums,
    final_norm_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_partial_sums
    
    # Load partial sums
    partial = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Sum all partial sums
    total = tl.sum(partial)
    
    # Compute sqrt for Frobenius norm
    norm = tl.sqrt(total)
    
    # Store final norm
    tl.store(final_norm_ptr, norm)


@triton.jit
def frobenius_normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm (scalar)
    norm = tl.load(norm_ptr)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor):
    """
    Compute Frobenius norm of tensor x using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Determine the number of blocks needed for first kernel
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # If n_blocks is 0 (empty tensor), handle special case
    if n_blocks == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    
    # Allocate partial sums tensor
    partial_sums = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    
    # Launch first kernel to compute partial sums of squares
    frobenius_norm_kernel[(n_blocks,)](
        x, n_elements, partial_sums,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Determine grid for final reduction kernel
    # We'll use a second kernel to sum the partial sums
    final_n_blocks = (n_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE
    if final_n_blocks == 0:
        final_n_blocks = 1
        
    final_norm = torch.empty(1, dtype=x.dtype, device=x.device)
    
    # Launch final kernel to compute final norm
    frobenius_norm_final_kernel[(final_n_blocks,)](
        partial_sums, n_blocks, final_norm,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return final_norm.squeeze()


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Apply Frobenius norm normalization to tensor x using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Compute Frobenius norm
    norm = triton_frobenius_norm(x)
    
    # Handle zero norm case to avoid division by zero
    # If norm is 0, return zeros (same as torch.norm behavior)
    if norm.item() == 0.0:
        return torch.zeros_like(x)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch normalization kernel
    frobenius_normalize_kernel[grid](
        x, norm, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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
        Applies Frobenius norm normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return triton_frobenius_normalize(x)