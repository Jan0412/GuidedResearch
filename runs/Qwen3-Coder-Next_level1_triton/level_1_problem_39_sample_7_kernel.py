import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows
    n_cols,  # Number of columns
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Computes L2 normalization: x / ||x||_2 along dimension 1.
    Each block processes one row, computing the norm and then normalizing.
    """
    row_start = tl.program_id(0)
    
    if row_start >= n_rows:
        return
    
    # Compute L2 norm for this row
    acc = tl.zeros([BLOCK_SIZE_COL], dtype=tl.float32)
    
    # Process columns in blocks to compute sum of squares
    for col_start in range(0, n_cols, BLOCK_SIZE_COL):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)
        mask = col_offsets < n_cols
        
        # Load values
        x_offsets = row_start * n_cols + col_offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Accumulate sum of squares
        acc += x * x
    
    # Reduce within the block
    norm_squared = tl.sum(acc, axis=0)
    
    # Compute sqrt
    norm = tl.sqrt(norm_squared + 1e-12)  # Add epsilon for numerical stability
    
    # Now normalize the row
    for col_start in range(0, n_cols, BLOCK_SIZE_COL):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)
        mask = col_offsets < n_cols
        
        # Load values
        x_offsets = row_start * n_cols + col_offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Normalize
        out = x / norm
        
        # Store result
        tl.store(out_ptr + x_offsets, out, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for L2 normalization.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L2 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_ROW = 1  # One block per row
    BLOCK_SIZE_COL = 1024  # Process columns in chunks of 1024
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the Triton kernel
    l2_norm_kernel[grid](
        x, 
        out, 
        n_rows, 
        n_cols,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).
            
        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        # Ensure tensor is on GPU and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Call the Triton-based L2 normalization
        return triton_l2_norm(x)