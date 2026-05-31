import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,  # Pointer to input tensor
    norm_ptr,  # Pointer to output norm (scalar)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    x_sq = x * x
    
    # Use reduction to sum all squared values
    # Note: We'll use a tree reduction pattern for efficiency
    acc = tl.sum(x_sq, axis=0, keep_dims=False)
    
    # Store partial sum in output buffer
    tl.store(norm_ptr + pid, acc)


@triton.jit
def frobenius_norm_final_kernel(
    partial_sums_ptr,  # Pointer to partial sums
    norm_ptr,  # Pointer to final norm result
    n_partial_sums,  # Number of partial sums
    BLOCK_SIZE: tl.constexpr,
):
    # Final reduction to get the total sum
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    x = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Final sum
    total = tl.sum(x, axis=0, keep_dims=False)
    
    # Take square root for Frobenius norm
    norm = tl.sqrt(total)
    
    tl.store(norm_ptr, norm)


@triton.jit
def normalize_kernel(
    x_ptr,  # Pointer to input tensor
    norm_ptr,  # Pointer to norm value
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    norm = tl.load(norm_ptr)
    
    # Avoid division by zero
    norm_safe = tl.where(norm > 0.0, norm, 1.0)
    
    out = x / norm_safe
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Applies Frobenius norm normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    
    # First kernel: compute partial sums of squared elements
    # Use multiple blocks for parallel reduction
    BLOCK_SIZE = 256
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create buffer for partial sums
    partial_sums = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    
    # Launch first kernel
    grid = (n_blocks,)
    frobenius_norm_kernel[grid](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute final norm from partial sums
    # For simplicity, we'll use a single block for final reduction if n_blocks <= 1024
    # Otherwise, we might need multiple passes, but for typical cases one more pass is sufficient
    final_n_blocks = (n_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE
    if final_n_blocks == 0:
        final_n_blocks = 1
    
    norm_buffer = torch.empty(1, device=x.device, dtype=torch.float32)
    
    # Launch final reduction kernel
    final_grid = (final_n_blocks,)
    frobenius_norm_final_kernel[final_grid](partial_sums, norm_buffer, n_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    # Third kernel: normalize the tensor
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    normalize_kernel[grid](x, norm_buffer, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
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