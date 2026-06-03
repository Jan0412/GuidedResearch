import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr
):
    # Calculate which row (or column depending on reduction dimension) this program handles
    row_idx = tl.program_id(0)
    
    # Calculate base pointers
    if DIM == 1:
        # Reducing along dimension 1 (rows)
        x_offset = row_idx * n_cols
        out_offset = row_idx
    else:
        # Reducing along dimension 0 (columns) - not implemented for this specific case
        # but kept for completeness
        x_offset = row_idx
        out_offset = row_idx
    
    # Initialize minimum with large value
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_cols if DIM == 1 else n_rows, BLOCK_SIZE):
        if DIM == 1:
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=float('inf'))
        else:
            # For reduction along dim=0, we need a different approach
            # This is simplified for the common case of dim=1
            break
            
        # Update minimum
        min_val = tl.minimum(min_val, x)
    
    # Parallel reduction within the block
    for stride in range(BLOCK_SIZE // 2, 0, stride // 2):
        if BLOCK_SIZE > 1:
            other = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) + stride, 
                          mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=float('inf'))
            min_val = tl.minimum(min_val, other)
    
    # Final reduction to get single minimum value
    if BLOCK_SIZE > 1:
        for stride in [32, 16, 8, 4, 2, 1]:
            if BLOCK_SIZE >= stride:
                other = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) % stride, 
                              mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=float('inf'))
                min_val = tl.minimum(min_val, other)
    
    # Store result
    if DIM == 1:
        tl.store(out_ptr + out_offset, min_val[0])
    else:
        tl.store(out_ptr + out_offset, min_val[0])


# Optimized kernel for min reduction along dimension 1
@triton.jit
def min_reduction_dim1_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Initialize minimum with positive infinity
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_idx * n_cols + offsets, mask=mask, other=float('inf'))
        min_val = tl.minimum(min_val, x)
    
    # Parallel reduction within the block
    # First, handle the case where BLOCK_SIZE might be larger than needed
    current_size = BLOCK_SIZE
    while current_size > 1:
        half = current_size // 2
        other = tl.load(x_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE) % half, 
                       mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=float('inf'))
        min_val = tl.minimum(min_val, other)
        current_size = half
    
    # Store result
    tl.store(out_ptr + row_idx, min_val[0])


# Better optimized kernel with proper reduction
@triton.jit
def min_reduction_kernel_v2(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Initialize minimum with positive infinity
    min_val = float('inf')
    
    # Process elements in this row
    for col_idx in range(0, n_cols):
        if col_idx < n_cols:
            val = tl.load(x_ptr + row_idx * n_cols + col_idx)
            min_val = tl.minimum(min_val, val)
    
    # Store result
    tl.store(out_ptr + row_idx, min_val)


# Efficient kernel with tiling for larger dimensions
@triton.jit
def min_reduction_kernel_tiled(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Initialize minimum with positive infinity
    min_val = float('inf')
    
    # Calculate start column for this block
    start_col = col_block * BLOCK_SIZE
    
    # Process elements in this tile
    for col_idx in range(start_col, min(start_col + BLOCK_SIZE, n_cols)):
        val = tl.load(x_ptr + row_idx * n_cols + col_idx)
        min_val = tl.minimum(min_val, val)
    
    # Store partial result to shared memory (using temporary buffer)
    # For simplicity, we'll use a different approach with atomic operations
    pass


# Final optimized version using block-level reduction
@triton.jit
def min_reduction_final(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Shared memory for reduction
    # tl.debug_print("Processing row: ", row_idx)
    
    # Initialize minimum with positive infinity
    min_val = float('inf')
    
    # Process all elements in the row
    for col_idx in range(0, n_cols):
        val = tl.load(x_ptr + row_idx * n_cols + col_idx)
        min_val = tl.minimum(min_val, val)
    
    # Store result
    tl.store(out_ptr + row_idx, min_val)


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton.
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
        Applies min reduction over the specified dimension to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        # Ensure tensor is on GPU and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Get input dimensions
        shape = x.shape
        dim = self.dim
        
        # Handle negative dimensions
        if dim < 0:
            dim = len(shape) + dim
        
        # Determine output shape and dimensions
        if dim == 1:
            n_rows = shape[0]
            n_cols = shape[1]
            # For higher dimensions, we need to reshape
            if len(shape) > 2:
                # Reshape to 2D: (batch, features) where features is product of remaining dims
                x_reshaped = x.view(n_rows, -1)
                n_cols = x_reshaped.shape[1]
            else:
                x_reshaped = x
            
            # Create output tensor
            out_shape = list(shape)
            out_shape[dim] = 1
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            
            # Calculate grid dimensions
            BLOCK_SIZE = 256
            grid = (n_rows,)
            
            # Launch kernel
            min_reduction_final[grid](x_reshaped, out.view(-1), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
            
            return out
        else:
            # For other dimensions, fall back to PyTorch (or implement more complex kernels)
            # This is a simplified implementation focusing on dim=1 which is common
            return torch.min(x, dim=dim)[0]