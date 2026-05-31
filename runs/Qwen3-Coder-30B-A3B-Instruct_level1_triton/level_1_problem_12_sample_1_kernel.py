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
    
    # Calculate offsets for this row
    a_offset = row_idx
    b_row_offset = row_idx * n_cols
    out_row_offset = row_idx * n_cols
    
    # Process this row in blocks
    for col_block in range(0, n_cols, BLOCK_SIZE):
        # Create column indices for this block
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load diagonal element
        a_val = tl.load(a_ptr + a_offset, mask=True)
        
        # Load B[row, :] elements
        b_vals = tl.load(b_ptr + b_row_offset + col_offsets, mask=mask, other=0.0)
        
        # Compute result: a_val * b_vals
        out_vals = a_val * b_vals
        
        # Store result
        tl.store(out_ptr + out_row_offset + col_offsets, out_vals, mask=mask)

def triton_diag_matmul(A, B):
    """
    Custom Triton kernel for diagonal matrix multiplication.
    Computes C = diag(A) @ B where A is 1D and B is 2D.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(B)
    
    # Get dimensions
    n_rows, n_cols = B.shape
    BLOCK_SIZE = 128
    
    # Grid configuration - one block per row
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A,
        B,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)