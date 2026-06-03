import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr,          # Input tensor pointer
    mask_ptr,       # Mask tensor pointer
    out_ptr,        # Output tensor pointer
    n_rows,         # Number of rows (batch dimension)
    n_cols,         # Number of columns (sequence dimension)
    stride_row,     # Stride for row dimension
    stride_col,     # Stride for column dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate base pointers for this row
    x_row_ptr = x_ptr + row_idx * stride_row
    mask_row_ptr = mask_ptr + row_idx * stride_row
    out_row_ptr = out_ptr + row_idx * stride_row
    
    # Cumulative sum accumulator
    cumsum = 0.0
    
    # Process columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_end = tl.minimum(col_start + BLOCK_SIZE, n_cols)
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load input values
        x_vals = tl.load(x_row_ptr + col_offsets * stride_col, mask=mask, other=0.0)
        mask_vals = tl.load(mask_row_ptr + col_offsets * stride_col, mask=mask, other=0).to(tl.int1)
        
        # Compute masked cumulative sum for this block
        block_cumsum = tl.zeros_like(x_vals)
        running_sum = 0.0
        
        # Sequential accumulation within the block
        for i in range(BLOCK_SIZE):
            if col_start + i < n_cols:
                if tl.load(mask_row_ptr + (col_start + i) * stride_col).to(tl.int1):
                    running_sum += tl.load(x_row_ptr + (col_start + i) * stride_col)
                block_cumsum = tl.where(
                    tl.arange(0, BLOCK_SIZE) == i,
                    running_sum,
                    block_cumsum
                )
        
        # Store results
        tl.store(out_row_ptr + col_offsets * stride_col, block_cumsum, mask=mask)


# Optimized version using warp-level operations and better parallelization
@triton.jit
def masked_cumsum_kernel_optimized(
    x_ptr,          # Input tensor pointer
    mask_ptr,       # Mask tensor pointer
    out_ptr,        # Output tensor pointer
    n_rows,         # Number of rows (batch dimension)
    n_cols,         # Number of columns (sequence dimension)
    stride_row,     # Stride for row dimension
    stride_col,     # Stride for column dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate base pointers for this row
    x_row_ptr = x_ptr + row_idx * stride_row
    mask_row_ptr = mask_ptr + row_idx * stride_row
    out_row_ptr = out_ptr + row_idx * stride_row
    
    # Initialize cumulative sum
    cumsum = 0.0
    
    # Process columns sequentially since cumsum has data dependencies
    for col_idx in range(n_cols):
        # Load mask and value
        current_mask = tl.load(mask_row_ptr + col_idx * stride_col).to(tl.int1)
        current_val = tl.load(x_row_ptr + col_idx * stride_col)
        
        # Update cumulative sum only if mask is True
        cumsum = tl.where(current_mask, cumsum + current_val, cumsum)
        
        # Store result
        tl.store(out_row_ptr + col_idx * stride_col, cumsum)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Compute masked cumulative sum along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        mask: Boolean mask tensor (same shape as x)
        dim: Dimension along which to compute cumulative sum
    
    Returns:
        Tensor with masked cumulative sum
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Convert mask to int for kernel processing
    mask_int = mask.to(torch.int32)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    if dim < 0:
        dim += x.dim()
    
    # Calculate strides
    strides = x.stride()
    stride_row = strides[dim] if dim < len(strides) else 1
    stride_col = 1 if dim == 0 else strides[0]
    
    # For 2D case specifically (batch, seq)
    if x.dim() == 2:
        n_rows, n_cols = x.shape
        stride_row = x.stride(0)
        stride_col = x.stride(1)
        
        # Set block size based on sequence length
        BLOCK_SIZE = min(128, n_cols)
        
        # Launch kernel - one block per row
        grid = (n_rows,)
        
        masked_cumsum_kernel_optimized[grid](
            x, mask_int, out,
            n_rows, n_cols,
            stride_row, stride_col,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Handle general case by reshaping to 2D
        # Move target dimension to position 1, reshape, process, then reshape back
        dims = list(range(x.dim()))
        if dim != 1:
            dims[1], dims[dim] = dims[dim], dims[1]
            x = x.permute(dims)
            mask = mask.permute(dims)
        
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        mask_flat = mask.reshape(mask.shape[0], -1)
        
        n_rows, n_cols = x_flat.shape
        stride_row = x_flat.stride(0)
        stride_col = x_flat.stride(1)
        
        BLOCK_SIZE = min(128, n_cols)
        grid = (n_rows,)
        
        out_flat = torch.empty_like(x_flat)
        
        masked_cumsum_kernel_optimized[grid](
            x_flat, mask_flat, out_flat,
            n_rows, n_cols,
            stride_row, stride_col,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        out = out_flat.reshape(original_shape)
        
        # Permute back if needed
        if dim != 1:
            inv_dims = [0] * len(dims)
            for i, d in enumerate(dims):
                inv_dims[d] = i
            out = out.permute(inv_dims)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using Triton kernels.
    
    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return triton_masked_cumsum(x, mask, self.dim)