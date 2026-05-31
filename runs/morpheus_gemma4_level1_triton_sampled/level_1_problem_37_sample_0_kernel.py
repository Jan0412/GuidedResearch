import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_squares_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program calculates the sum of squares for a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements and compute squares
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sq = x * x
    
    # Reduce the block to a single sum
    sum_sq = tl.sum(sq, axis=0)
    
    # Store the partial sum in the output buffer
    tl.store(out_ptr + pid, sum_sq)

@triton.jit
def div_kernel(
    x_ptr, 
    out_ptr, 
    norm, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements for division
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Divide by the scalar norm
    out = x / norm
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm_normalize(x: torch.Tensor):
    """
    Optimized Frobenius norm normalization using Triton.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Ensure input is contiguous for linear indexing
    x = x.contiguous()
    n_elements = x.numel()
    
    # Block size for kernel execution
    BLOCK_SIZE = 1024
    
    # 1. Compute Sum of Squares
    grid_sum = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(grid_sum, device=x.device, dtype=torch.float32)
    
    sum_squares_kernel[grid_sum](
        x, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction of partial sums to get the global Frobenius norm
    # Using torch.sum on the small partial_sums buffer is efficient
    total_sum_sq = torch.sum(partial_sums)
    norm = torch.sqrt(total_sum_sq)
    
    # 2. Element-wise Division
    out = torch.empty_like(x)
    div_kernel[grid_sum](
        x, 
        out, 
        norm, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom Triton kernels.
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
        return triton_frobenius_norm_normalize(x)