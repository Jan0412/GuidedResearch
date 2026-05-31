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
    tl.atomic_add(norm_ptr, tl.sum(x_squared), sem="acquire_release")

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
    """Compute Frobenius norm using Triton"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Use a single element tensor to store the sum of squares
    norm_tensor = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    frobenius_norm_kernel[grid](x, norm_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Get the final norm value
    norm_val = torch.sqrt(norm_tensor.item())
    
    return norm_val

def triton_normalize(x: torch.Tensor, norm_val: float):
    """Normalize tensor using Triton"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    normalize_kernel[grid](x, norm_val, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    Optimized with custom Triton kernels.
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
        # Compute Frobenius norm using Triton
        norm = triton_frobenius_norm(x)
        
        # Normalize using Triton
        return triton_normalize(x, norm)