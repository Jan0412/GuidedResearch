import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr, 
    out_ptr, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_x_ptr = x_ptr + row_idx * n_cols
    row_out_ptr = out_ptr + row_idx * n_cols

    # --- Pass 1: Compute the sum of squares (L2 norm squared) ---
    sum_sq = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load block of data, masking out-of-bounds elements
        val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(val * val, axis=0)

    # Compute the inverse of the norm: 1 / sqrt(sum_sq)
    # We allow division by zero to result in inf/nan to match PyTorch behavior
    inv_norm = 1.0 / tl.sqrt(sum_sq)

    # --- Pass 2: Normalize the row and store the result ---
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        # Multiply by the precomputed inverse norm
        normalized_val = val * inv_norm
        tl.store(row_out_ptr + offsets, normalized_val, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along the last dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)

    # BLOCK_SIZE is the number of elements processed per iteration within a row.
    # For dim=65535, 1024 is a reasonable choice.
    BLOCK_SIZE = 1024
    
    # Grid is 1D, with one program per row.
    grid = (batch_size,)

    l2_norm_kernel[grid](
        x, 
        out, 
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
        Applies L2 normalization to the input tensor along dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        # The original model performs normalization along dim=1.
        # Our Triton kernel is designed for 2D tensors normalized across the second dimension.
        return triton_l2_norm(x)