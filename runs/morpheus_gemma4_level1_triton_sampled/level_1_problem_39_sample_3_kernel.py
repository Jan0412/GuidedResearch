import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Calculate the base pointer for the current row
    row_start_ptr = x_ptr + row_idx * dim
    out_row_start_ptr = out_ptr + row_idx * dim

    # First pass: Compute the sum of squares for the L2 norm
    sum_sq = 0.0
    for k in range(0, dim, BLOCK_SIZE):
        cols = k + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        # Load elements and accumulate their squares
        val = tl.load(row_start_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(val * val, axis=0)

    # Compute the inverse norm (1 / sqrt(sum of squares))
    # We allow division by zero to produce inf/nan to match PyTorch's behavior
    inv_norm = 1.0 / tl.sqrt(sum_sq)

    # Second pass: Normalize each element and store the result
    for k in range(0, dim, BLOCK_SIZE):
        cols = k + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        val = tl.load(row_start_ptr + cols, mask=mask, other=0.0)
        tl.store(out_row_start_ptr + cols, val * inv_norm, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Wrapper for the Triton L2 normalization kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the input is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    
    # Prepare the output tensor
    out = torch.empty_like(x)
    
    # Block size for processing columns within a row
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)
    
    l2_norm_kernel[grid](
        x, 
        out, 
        batch_size, 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x)