import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (elements per row to reduce)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row (one reduction)
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    x_offset = row_idx * n_cols
    
    # Initialize minimum with a large value (FLT_MAX for FP32 is ~3.4e38)
    min_val = tl.full((BLOCK_SIZE,), 3.4028235e+38, dtype=tl.float32)
    
    # Process in blocks to find minimum
    for start in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=3.4028235e+38)
        
        # Update minimum
        min_val = tl.minimum(min_val, x)
    
    # Final reduction across the block dimension to get single minimum value
    # We use a tree reduction approach
    for i in range(BLOCK_SIZE // 2):
        min_val = tl.minimum(min_val, tl.roll(min_val, 1 << i, 0))
    
    # Store result (only first lane has the correct value)
    if tl.program_id(0) < n_rows:
        tl.store(out_ptr + row_idx, min_val[0])


def triton_min_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs min reduction along specified dimension using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after reduction
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Handle 2D case specifically for simplicity (common case)
    if len(shape) == 3:
        # Reshape to 2D if needed
        if dim == 1:
            # (batch, dim1, dim2) -> reduce dim1 -> (batch, dim2)
            batch_size, dim1, dim2 = shape
            x_2d = x.transpose(1, 2).contiguous().view(batch_size * dim2, dim1)
            out_shape = (batch_size, dim2)
        elif dim == 2:
            # (batch, dim1, dim2) -> reduce dim2 -> (batch, dim1)
            batch_size, dim1, dim2 = shape
            x_2d = x.view(batch_size * dim1, dim2)
            out_shape = (batch_size, dim1)
        else:
            raise NotImplementedError("Only supports dim=1 or dim=2 for 3D tensors")
        
        # Prepare output tensor
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        
        # Calculate grid dimensions
        n_rows = x_2d.shape[0]
        n_cols = x_2d.shape[1]
        BLOCK_SIZE = 256  # Tunable parameter
        
        grid = (n_rows,)
        
        # Launch kernel
        min_reduction_kernel[grid](x_2d, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        return out
    elif len(shape) == 2:
        # 2D case: (batch, features)
        if dim == 1:
            batch_size, n_cols = shape
            out = torch.empty(batch_size, dtype=x.dtype, device=x.device)
            
            n_rows = batch_size
            BLOCK_SIZE = 256
            
            grid = (n_rows,)
            
            min_reduction_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
            return out
        else:
            raise NotImplementedError("Only supports dim=1 for 2D tensors")
    else:
        raise NotImplementedError("Only supports 2D and 3D tensors")


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernel.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)