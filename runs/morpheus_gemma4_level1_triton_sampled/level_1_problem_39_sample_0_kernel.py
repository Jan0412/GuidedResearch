import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    dim,            # Dimension along which to normalize (dim=1)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Calculate the base pointer for the current row
    row_start_ptr = x_ptr + row_idx * dim
    out_start_ptr = out_ptr + row_idx * dim

    # First pass: Compute the sum of squares for the row
    sum_sq = 0.0
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load a block of elements
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x)

    # Compute the inverse of the L2 norm
    # Adding a small epsilon to avoid division by zero
    inv_norm = 1.0 / tl.sqrt(sum_sq + 1e-12)

    # Second pass: Divide each element by the norm and store the result
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load the same block again
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Normalize and store
        tl.store(out_start_ptr + offsets, x * inv_norm, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along the second dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)

    # BLOCK_SIZE is the number of elements processed per iteration within a row
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)

    # Launch the Triton kernel
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
        Applies L2 normalization to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)