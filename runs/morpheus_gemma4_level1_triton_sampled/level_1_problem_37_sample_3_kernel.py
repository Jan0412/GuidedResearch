import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_sq_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program computes a partial sum of squares for its block
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute square
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local_sum = tl.sum(x * x, axis=0)
    
    # Atomic add to accumulate global sum of squares
    tl.atomic_add(out_ptr, local_sum)

@triton.jit
def scale_kernel(
    x_ptr, 
    out_ptr, 
    inv_norm, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program scales a block of the tensor by the inverse norm
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * inv_norm
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm_normalization(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA"
    
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    n_elements = x.numel()
    
    # 1. Compute sum of squares
    # We use a single-element tensor to store the global sum
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE_RED = 1024
    grid_red = ((n_elements + BLOCK_SIZE_RED - 1) // BLOCK_SIZE_RED,)
    
    sum_sq_kernel[grid_red](
        x, 
        sum_sq, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE_RED
    )
    
    # 2. Compute the Frobenius norm and its reciprocal
    # norm = sqrt(sum(x^2))
    norm = torch.sqrt(sum_sq)
    
    # Handle potential division by zero
    inv_norm = 1.0 / (norm + 1e-12)
    inv_norm_val = inv_norm.item()
    
    # 3. Scale the original tensor
    out = torch.empty_like(x)
    BLOCK_SIZE_SCALE = 1024
    grid_scale = ((n_elements + BLOCK_SIZE_SCALE - 1) // BLOCK_SIZE_SCALE,)
    
    scale_kernel[grid_scale](
        x, 
        out, 
        inv_norm_val, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE_SCALE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using custom Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use the custom Triton implementation for speedup
        return triton_frobenius_norm_normalization(x)