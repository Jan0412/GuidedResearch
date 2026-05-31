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
    # Compute squared values and accumulate
    x_sq = x * x
    # Use atomic operation to accumulate sum across all blocks
    tl.atomic_add(out_ptr, tl.sum(x_sq, axis=0), sem="acq_rel")

@triton.jit
def divide_kernel(
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

def triton_frobenius_norm(x: torch.Tensor):
    """Compute Frobenius norm using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Flatten the tensor to compute norm
    flat_x = x.view(-1)
    n_elements = flat_x.numel()
    
    # Allocate buffer for accumulation
    norm_buffer = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Compute sum of squares
    norm_kernel[grid](flat_x, norm_buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Take square root to get norm
    norm = torch.sqrt(norm_buffer)
    return norm

def triton_divide(x: torch.Tensor, norm: torch.Tensor):
    """Divide tensor by norm using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    flat_x = x.view(-1)
    n_elements = flat_x.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Divide each element by norm
    divide_kernel[grid](flat_x, norm, out.view(-1), n_elements, BLOCK_SIZE=BLOCK_SIZE)
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
        # Compute Frobenius norm using Triton
        norm = triton_frobenius_norm(x)
        
        # Normalize using Triton
        return triton_divide(x, norm)