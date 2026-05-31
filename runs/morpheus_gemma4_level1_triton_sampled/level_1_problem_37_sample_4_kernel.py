import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_squares_kernel(
    x_ptr, 
    sum_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel to calculate the sum of squares of all elements in the tensor.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements and compute local sum of squares
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local_sum = tl.sum(x * x, axis=0)
    
    # Atomically add the local sum to the global sum
    tl.atomic_add(sum_ptr, local_sum)

@triton.jit
def norm_div_kernel(
    x_ptr, 
    norm_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel to divide every element of the tensor by the computed Frobenius norm.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements and the scalar norm
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    norm = tl.load(norm_ptr)
    
    # Perform division and store result
    tl.store(out_ptr + offsets, x / norm, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Wrapper function to orchestrate the Frobenius norm normalization using Triton kernels.
    """
    # Ensure input is contiguous on GPU
    x = x.contiguous()
    n_elements = x.numel()
    
    # 1. Compute sum of squares
    # Initialize a single-element tensor to hold the global sum
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE_SUM = 1024
    grid_sum = ((n_elements + BLOCK_SIZE_SUM - 1) // BLOCK_SIZE_SUM,)
    sum_squares_kernel[grid_sum](x, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE_SUM)
    
    # 2. Compute the Frobenius norm (sqrt of sum of squares)
    norm = torch.sqrt(sum_sq)
    
    # 3. Divide the input tensor by the norm
    out = torch.empty_like(x)
    BLOCK_SIZE_DIV = 1024
    grid_div = ((n_elements + BLOCK_SIZE_DIV - 1) // BLOCK_SIZE_DIV,)
    norm_div_kernel[grid_div](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE_DIV)
    
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
        Applies Frobenius norm normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return triton_frobenius_norm(x)