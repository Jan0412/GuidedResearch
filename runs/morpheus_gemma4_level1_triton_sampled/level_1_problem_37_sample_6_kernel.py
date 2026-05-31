import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_squares_kernel(
    x_ptr, 
    sum_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares for the block
    sq = x * x
    block_sum = tl.sum(sq, axis=0)
    
    # Atomically add the block sum to the global sum scalar
    tl.atomic_add(sum_ptr, block_sum)

@triton.jit
def normalize_kernel(
    x_ptr, 
    out_ptr, 
    inv_norm, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply by the pre-computed inverse of the Frobenius norm
    out = x * inv_norm
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Custom Triton implementation of Frobenius norm normalization.
    1. Computes the sum of squares using atomic additions.
    2. Computes the inverse of the square root of that sum.
    3. Performs element-wise multiplication.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    
    # 1. Sum of Squares
    # We use a single-element tensor to store the global sum of squares
    sum_sq = torch.zeros((1,), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid_sum = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    sum_squares_kernel[grid_sum](
        x, sum_sq, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 2. Compute inverse norm in PyTorch (scalar operation)
    # Frobenius norm = sqrt(sum of squares)
    norm = torch.sqrt(sum_sq)
    # Avoid division by zero
    inv_norm = 1.0 / (norm + 1e-12)
    inv_norm_val = inv_norm.item()
    
    # 3. Element-wise Normalization
    out = torch.empty_like(x)
    grid_norm = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    normalize_kernel[grid_norm](
        x, out, inv_norm_val, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
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
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied.
        """
        return triton_frobenius_norm(x)