import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    diag_ptr,  # Pointer to diagonal elements (1D tensor of size N)
    mat_ptr,   # Pointer to matrix B (2D tensor of shape N x M)
    out_ptr,   # Pointer to output matrix (2D tensor of shape N x M)
    N,         # Number of rows
    M,         # Number of columns
    stride_diag,  # Stride for diagonal tensor (typically 1 for contiguous)
    stride_mat,   # Stride for matrix B (typically M for contiguous rows)
    stride_out,   # Stride for output matrix (typically M for contiguous rows)
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the matrix multiplication
    # Since we're doing diag(A) @ B, each output row i is simply A[i] * B[i]
    row_idx = tl.program_id(0)
    
    if row_idx >= N:
        return
    
    # Compute pointers to current row in B and output
    mat_row_start = mat_ptr + row_idx * stride_mat
    out_row_start = out_ptr + row_idx * stride_out
    
    # Load the diagonal element A[row_idx]
    diag_val = tl.load(diag_ptr + row_idx * stride_diag)
    
    # Process the row in blocks
    for col_start in range(0, M, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < M
        
        # Load B[row_idx, col_start:col_start+BLOCK_SIZE]
        b_vals = tl.load(mat_row_start + col_offsets, mask=mask, other=0.0)
        
        # Compute A[row_idx] * B[row_idx, col_start:col_start+BLOCK_SIZE]
        out_vals = diag_val * b_vals
        
        # Store result to output
        tl.store(out_row_start + col_offsets, out_vals, mask=mask)


def triton_diag_matmul(diag: torch.Tensor, mat: torch.Tensor):
    """
    Computes C = diag(diag) @ mat using a custom Triton kernel.
    
    Args:
        diag: 1D tensor of shape (N,) representing diagonal elements
        mat: 2D tensor of shape (N, M)
    
    Returns:
        2D tensor of shape (N, M)
    """
    assert diag.is_cuda and mat.is_cuda, "Tensors must be on CUDA."
    assert diag.dim() == 1, "diag must be 1D"
    assert mat.dim() == 2, "mat must be 2D"
    assert diag.shape[0] == mat.shape[0], f"diag length {diag.shape[0]} must match mat rows {mat.shape[0]}"
    
    # Ensure inputs are contiguous
    diag = diag.contiguous()
    mat = mat.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(mat)
    
    N, M = mat.shape
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Grid: one block per row
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        diag, mat, out,
        N, M,
        diag.stride(0), mat.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for diag(A) @ B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)