import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU parameters (fixed constants)
    alpha = 1.050700987119216
    lambda_val = 1.673263242354377
    
    # Compute SELU: for x > 0: alpha * x, for x <= 0: alpha * lambda * (exp(x) - 1)
    # Use tl.where for efficient conditional without branching
    pos_part = alpha * x
    neg_part = alpha * lambda_val * (tl.exp(x) - 1.0)
    out = tl.where(x > 0, pos_part, neg_part)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Apply SELU activation using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimized block size for modern GPUs
    
    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation using optimized Triton kernel.
        """
        return triton_selu(x)