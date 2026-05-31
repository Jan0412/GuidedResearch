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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_squared = x * x
    tl.atomic_add(norm_ptr, tl.sum(x_squared), sem="relaxed")

@triton.jit
def normalize_kernel(
    x_ptr,
    norm_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / norm_val
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """Compute Frobenius norm using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Allocate memory for norm squared
    norm_squared = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    # Calculate grid
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel to compute sum of squares
    frobenius_norm_kernel[grid](x, norm_squared, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute final norm
    norm = torch.sqrt(norm_squared)
    return norm

def triton_normalize(x: torch.Tensor, norm: torch.Tensor):
    """Normalize tensor using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate grid
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel to perform normalization
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
        Applies Frobenius norm normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Compute Frobenius norm using Triton kernel
        norm = triton_frobenius_norm(x)
        
        # Normalize using Triton kernel
        return triton_normalize(x, norm)