import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def norm_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute squared elements and sum them
    x_sq = x * x
    tl.atomic_add(out_ptr, tl.sum(x_sq, axis=0), sem="acquire_release")

@triton.jit
def div_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    norm = tl.load(norm_ptr)
    out = x / norm
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_norm(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Compute squared sum using atomic operations
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Initialize accumulator
    norm_sq = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    norm_kernel[grid](x, norm_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Take square root to get Frobenius norm
    norm = torch.sqrt(norm_sq)
    return norm

def triton_divide(x: torch.Tensor, norm: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    div_kernel[grid](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = triton_norm(x)
        return triton_divide(x, norm)