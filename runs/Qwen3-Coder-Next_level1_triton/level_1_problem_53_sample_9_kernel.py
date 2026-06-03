import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of columns in the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    row_idx = tl.program_id(0)
    # Calculate offset to the start of this row
    row_start = row_idx * n_cols

    # Initialize min value with the largest possible float32 value
    min_val = tl.full([1], float("inf"), dtype=tl.float32)

    # Iterate over the row in chunks of BLOCK_SIZE
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float("inf"))
        
        # Compute minimum
        min_val = tl.minimum(min_val, tl.min(x, axis=0))

    # Store the result
    tl.store(out_ptr + row_idx, min_val)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of min reduction over a specified dimension.
    Optimized for FP32 precision.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape information
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Move reduction dimension to the last position for easier handling
    if dim != len(shape) - 1:
        perm = list(range(len(shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm)
        shape = x.shape
    
    # Output shape is all dimensions except the last one
    out_shape = shape[:-1]
    n_rows = 1
    for s in out_shape:
        n_rows *= s
    n_cols = shape[-1]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Use a reasonable block size - tuned for FP32
    BLOCK_SIZE = 256
    
    # Grid is 1D: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    min_reduction_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for min reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min(x, self.dim)