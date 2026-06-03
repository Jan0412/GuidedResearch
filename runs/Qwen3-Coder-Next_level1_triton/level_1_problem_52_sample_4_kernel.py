import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    X,  # Input pointer
    OUT,  # Output pointer
    stride_x_batch, stride_x_dim1, stride_x_dim2,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for argmin along the last dimension (dim=-1).
    Each program handles one row of the input tensor.
    """
    # Program ID corresponds to the row index
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    row_start = row_idx * stride_x_batch + 0 * stride_x_dim1 + 0 * stride_x_dim2
    
    # Initialize minimum value and index
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    
    # Process in chunks of BLOCK_SIZE
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        # Load values
        x_ptr = row_start + offset * stride_x_dim2
        x = tl.load(x_ptr, mask=mask, other=float('inf')).to(tl.float32)
        
        # Create indices for this block
        indices = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Update minimum if we find a smaller value
        is_smaller = (x < min_val) & mask
        min_val = tl.where(is_smaller, x, min_val)
        min_idx = tl.where(is_smaller, indices, min_idx)
    
    # Final reduction to find the single minimum across all blocks
    # We'll do a sequential reduction for simplicity
    final_min_val = min_val[0]
    final_min_idx = min_idx[0]
    
    for i in range(1, BLOCK_SIZE):
        curr_val = min_val[i]
        curr_idx = min_idx[i]
        is_smaller = curr_val < final_min_val
        final_min_val = tl.where(is_smaller, curr_val, final_min_val)
        final_min_idx = tl.where(is_smaller, curr_idx, final_min_idx)
    
    # Store result
    out_ptr = OUT + row_idx * stride_x_batch
    tl.store(out_ptr, final_min_idx.to(tl.int64))


class TritonArgminFunction(torch.autograd.Function):
    """Custom autograd function for argmin operation."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get tensor shapes
        shape = x.shape
        ndim = len(shape)
        
        # Normalize dimension
        if dim < 0:
            dim = ndim + dim
        
        # For simplicity, we'll handle dim=1 case specifically
        # If dim is not the last dimension, we'll transpose
        if dim != ndim - 1:
            # Permute to move dim to last position
            perm = list(range(ndim))
            perm.pop(dim)
            perm.append(dim)
            x = x.permute(perm).contiguous()
            original_perm = perm
        else:
            original_perm = None
        
        shape = x.shape
        n_rows = 1
        for i in range(len(shape) - 1):
            n_rows *= shape[i]
        n_cols = shape[-1]
        
        # Prepare output tensor
        out_shape = list(shape[:-1])
        out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
        
        # Set up strides
        stride_x_batch = x.stride(0) if len(shape) > 1 else 0
        stride_x_dim1 = x.stride(1) if len(shape) > 2 else 0
        stride_x_dim2 = x.stride(-1)
        
        # Determine block size (use a reasonable default)
        BLOCK_SIZE = 128
        
        # Grid: one block per row
        grid = (n_rows,)
        
        # Launch kernel
        argmin_kernel[grid](
            x, out,
            stride_x_batch=stride_x_batch,
            stride_x_dim1=stride_x_dim1,
            stride_x_dim2=stride_x_dim2,
            n_rows=n_rows, n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # If we permuted, permute back
        if original_perm is not None:
            # Invert the permutation
            inv_perm = [0] * len(original_perm)
            for i, p in enumerate(original_perm):
                inv_perm[p] = i
            out = out.permute(inv_perm).contiguous()
        
        return out


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper function for Triton argmin."""
    return TritonArgminFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using custom Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)