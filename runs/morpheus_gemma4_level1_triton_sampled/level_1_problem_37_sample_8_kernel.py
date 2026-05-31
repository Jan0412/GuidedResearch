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
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute square
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sq = x * x
    
    # Sum the squares within the block
    local_sum = tl.sum(sq, axis=0)
    
    # Atomically add the local sum to the global accumulator
    tl.atomic_add(out_ptr, local_sum)

@triton.jit
def norm_div_kernel(
    x_ptr, 
    out_ptr, 
    norm_val, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and divide by the pre-computed norm
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x / norm_val
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Triton implementation of Frobenius norm normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous().float()
    n_elements = x.numel()
    
    # 1. Compute Sum of Squares
    # We use a single-element tensor to accumulate the global sum
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    sum_squares_kernel[grid](
        x, 
        sum_sq, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 2. Compute the Frobenius norm (sqrt of sum of squares)
    norm_val = torch.sqrt(sum_sq).item()
    
    # 3. Element-wise division
    out = torch.empty_like(x)
    norm_div_kernel[grid](
        x, 
        out, 
        norm_val, 
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
        return triton_frobenius_norm(x)