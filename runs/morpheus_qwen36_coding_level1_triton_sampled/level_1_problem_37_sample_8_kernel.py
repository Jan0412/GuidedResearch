import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_sq_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes the sum of squares of elements in blocks.
    Each program handles a block of elements and computes a partial sum.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squares
    sq = x * x
    
    # Compute partial sum for this block
    # Use tl.reduce with tl.add to sum along the block dimension
    partial_sum = tl.reduce(sq, axis=0, combine_fn=tl.add)
    
    # Store the partial sum
    tl.store(out_ptr + tl.program_id(0), partial_sum)


@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    norm,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalizes the input tensor by dividing each element by the norm.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize
    out = x / norm
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Frobenius norm normalization using custom Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Step 1: Compute partial sums of squares
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.zeros(n_blocks, dtype=torch.float32, device=x.device)
    
    sum_sq_kernel[lambda meta: (n_blocks,)](
        x_ptr=x,
        out_ptr=partial_sums,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 2: Reduce partial sums to get total sum of squares
    total_sum_sq = torch.sum(partial_sums).item()
    norm = torch.sqrt(torch.tensor(total_sum_sq, dtype=torch.float32, device=x.device))
    
    # Step 3: Normalize the tensor
    out = torch.empty_like(x)
    normalize_kernel[lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        norm=norm,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)