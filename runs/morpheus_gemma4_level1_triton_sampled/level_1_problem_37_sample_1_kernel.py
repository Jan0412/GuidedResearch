import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_sq_kernel(
    x_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute the sum of squares of all elements in a tensor.
    Uses atomic addition to accumulate the global sum.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements and compute local sum of squares
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local_sum = tl.sum(x * x, axis=0)
    
    # Atomically add the local sum to the global accumulator
    tl.atomic_add(sum_ptr, local_sum)

@triton.jit
def div_kernel(
    x_ptr,
    out_ptr,
    norm_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to divide every element in a tensor by a scalar value.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division by the Frobenius norm
    out = x / norm_val
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Wrapper function that orchestrates the Triton kernels to perform 
    Frobenius norm normalization.
    """
    # Ensure the tensor is contiguous on GPU for Triton access
    x = x.contiguous()
    n_elements = x.numel()
    
    # Buffer to store the global sum of squares (FP32)
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Step 1: Compute the sum of squares of all elements
    sum_sq_kernel[grid](x, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Calculate the Frobenius norm (scalar)
    # We use torch.sqrt on the single-element tensor and extract the value
    norm_val = torch.sqrt(sum_sq).item()
    
    # Step 3: Divide every element by the norm
    out = torch.empty_like(x)
    div_kernel[grid](x, out, norm_val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
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
            torch.Tensor: Output tensor with Frobenius norm normalization applied.
        """
        return triton_frobenius_norm(x)