import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start = row_idx * n_cols
    
    # Initialize max with lowest possible value for FP32
    max_val = -tl.libdevice.inf(tl.float32)
    
    # Iterate over columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-tl.libdevice.inf(tl.float32))
        
        # Compute max
        max_val = tl.maximum(max_val, tl.max(x))
    
    # Store result
    tl.store(out_ptr + row_idx, max_val)


def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs max reduction over the specified dimension using Triton.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Determine the shape after reduction
    shape = list(x.shape)
    if dim < 0:
        dim += len(shape)
    
    # Compute output shape and sizes
    output_shape = shape[:dim] + shape[dim+1:]
    n_rows = 1
    for s in output_shape:
        n_rows *= s
    n_cols = shape[dim]
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256  # Tunable parameter
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    max_reduction_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton kernel.
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
        Applies Max reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max(x, self.dim)