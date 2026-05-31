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
    """
    Computes the sum of squares for Frobenius norm calculation.
    """
    # Each program handles a portion of the input
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    
    # Store partial sum
    tl.atomic_add(norm_ptr, sum_sq)


@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalizes the input tensor by dividing by the precomputed norm.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm (scalar)
    norm = tl.load(norm_ptr)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Applies Frobenius norm normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    # Grid for reduction
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare norm buffer (single element on GPU)
    norm_buffer = torch.zeros(1, device=x.device, dtype=x.dtype)
    
    # First kernel: compute sum of squares
    frobenius_norm_kernel[n_blocks](x, norm_buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute sqrt on GPU to get the actual norm
    norm = torch.sqrt(norm_buffer[0])
    
    # Grid for normalization
    n_blocks_norm = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Second kernel: normalize
    normalize_kernel[n_blocks_norm](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
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