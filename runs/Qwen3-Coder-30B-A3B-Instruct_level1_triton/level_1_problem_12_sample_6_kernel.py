import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one row
    if row_idx >= n_rows:
        return
    
    # Load the diagonal element for this row
    a_val = tl.load(a_ptr + row_idx)
    
    # Process columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create column offsets
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load B[row, col] values
        b_vals = tl.load(b_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
        
        # Perform element-wise multiplication: a_val * B[row, col]
        out_vals = a_val * b_vals
        
        # Store results
        tl.store(out_ptr + row_idx * n_cols + col_offsets, out_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication C = diag(A) @ B using Triton kernel.
        
        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        # Ensure inputs are on GPU and contiguous
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        
        # Prepare output tensor
        C = torch.empty_like(B)
        
        # Get dimensions
        N, M = B.shape
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid size
        grid = (N, 1, 1)
        
        # Launch kernel
        diag_matmul_kernel[grid](
            A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return C