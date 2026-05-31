import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_squares_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute local sum of squares
    local_sum = tl.sum(x * x, axis=0)
    # Atomically add the local sum to the global accumulator
    tl.atomic_add(out_ptr, local_sum)

@triton.jit
def scale_kernel(
    x_ptr, 
    norm_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load the pre-computed norm (scalar)
    norm = tl.load(norm_ptr)
    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Divide by norm and store the result
    tl.store(out_ptr + offsets, x / norm, mask=mask)

def triton_frobenius_norm_normalization(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for the kernel
    x = x.contiguous()
    n_elements = x.numel()
    
    # 1. Compute sum of squares
    # We use a single scalar tensor to accumulate the sum of squares
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid_sum = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    sum_squares_kernel[grid_sum](
        x, 
        sum_sq, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute the Frobenius norm: sqrt(sum of squares)
    norm = torch.sqrt(sum_sq)
    
    # 2. Scale the tensor by the norm
    out = torch.empty_like(x)
    grid_scale = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    scale_kernel[grid_scale](
        x, 
        norm, 
        out, 
        n_elements, 
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
        Applies Frobenius norm normalization to the input tensor using custom Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return triton_frobenius_norm_normalization(x)