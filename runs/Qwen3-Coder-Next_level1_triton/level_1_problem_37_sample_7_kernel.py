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
    """Compute the Frobenius norm of a tensor using parallel reduction."""
    # Calculate the global index of the thread
    pid = tl.program_id(0)
    # Create offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load elements and compute squared values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_squared = x * x
    
    # Initialize accumulator
    sum_val = tl.sum(x_squared, axis=0)
    
    # Perform reduction across blocks using atomic operations
    # For simplicity in this kernel, we'll use a single block approach for small tensors
    # or a two-pass approach for larger tensors
    # Here we'll do a simple approach: each block computes partial sum, then we do final reduction on CPU/GPU
    # But for simplicity, let's use a single block for the main computation if possible
    
    # Store partial sum
    tl.store(norm_ptr + pid, sum_val)


@triton.jit
def normalize_frobenius_kernel(
    x_ptr,
    norm,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize tensor by Frobenius norm."""
    # Calculate the global index of the thread
    pid = tl.program_id(0)
    # Create offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Apply Frobenius norm normalization using Triton kernels.
    
    Args:
        x (torch.Tensor): Input tensor of arbitrary shape
        
    Returns:
        torch.Tensor: Output tensor with Frobenius norm normalization applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # First pass: compute sum of squares
    # For simplicity, we'll use a single kernel that processes the entire tensor
    # and accumulates to a single scalar using a reduction approach
    
    # Use a single block for the reduction if the tensor is small enough
    # Otherwise, use multiple blocks and then do final reduction on CPU
    num_blocks = min((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1024)
    
    # Create a buffer for partial sums
    partial_sums = torch.zeros(num_blocks, device=x.device, dtype=torch.float32)
    
    # Launch kernel to compute partial sums of squares
    frobenius_norm_kernel[(num_blocks,)](
        x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums[:num_blocks])
    norm = torch.sqrt(total_sum)
    
    # Avoid division by zero
    norm = torch.where(norm == 0, torch.tensor(1.0, device=x.device), norm)
    
    # Second pass: normalize the tensor
    num_blocks_out = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    normalize_frobenius_kernel[(num_blocks_out,)](
        x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
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