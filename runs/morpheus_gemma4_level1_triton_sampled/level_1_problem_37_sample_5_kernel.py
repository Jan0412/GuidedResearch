import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_sq_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to scalar output for sum of squares
    n_elements, # Total elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute local sum of squares for this block
    local_sum_sq = tl.sum(x * x)
    
    # Atomically add the local sum to the global sum of squares
    tl.atomic_add(out_ptr, local_sum_sq)

@triton.jit
def normalize_kernel(
    x_ptr,      # Pointer to input tensor
    sum_sq_ptr, # Pointer to scalar sum of squares
    out_ptr,    # Pointer to output tensor
    n_elements, # Total elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the global sum of squares and compute the Frobenius norm
    sum_sq = tl.load(sum_sq_ptr)
    norm = tl.sqrt(sum_sq)

    # Load input values and normalize
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Division by norm (handles norm=0 by producing inf/nan, matching PyTorch behavior)
    out = x / norm
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Triton-optimized Frobenius norm normalization.
    """
    # Store original shape to restore it at the end
    orig_shape = x.shape
    
    # Ensure input is contiguous and FP32 on GPU
    x_flat = x.contiguous().view(-1).float()
    n_elements = x_flat.numel()
    
    # Prepare output tensor and a scalar for the global sum of squares
    out_flat = torch.empty_like(x_flat)
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Pass 1: Compute the total sum of squares (reduction)
    sum_sq_kernel[grid](x_flat, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Pass 2: Divide every element by the square root of the sum of squares
    normalize_kernel[grid](x_flat, sum_sq, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape back to original dimensions
    return out_flat.view(orig_shape)

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