import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of elements in the reduction dimension
    n_rows,  # Number of rows (other dimensions combined)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    row_start = row_idx * n_cols
    
    # Accumulator for the sum
    sum_ = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data with mask
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_ += tl.sum(x, axis=0)
    
    # Compute mean
    mean = sum_ / n_cols
    
    # Store result
    tl.store(out_ptr + row_idx, mean)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute mean along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        
    Returns:
        Output tensor with the specified dimension reduced
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape information
    shape = x.shape
    dim = dim if dim >= 0 else len(shape) + dim
    
    # Calculate sizes
    n_rows = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    
    n_cols = shape[dim]
    
    # Create output shape
    out_shape = list(shape)
    del out_shape[dim]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Determine block size (tuned for FP32)
    BLOCK_SIZE = 1024  # Good default for mean reduction
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Launch kernel
    mean_kernel[grid](x, out, n_cols, n_rows, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension
    using Triton kernel.
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
        Reduces the input tensor along the specified dimension by taking the mean
        using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)