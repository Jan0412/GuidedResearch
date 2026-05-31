import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal vector A (shape: N)
    b_ptr,  # Pointer to matrix B (shape: N x M)
    c_ptr,  # Pointer to output matrix C (shape: N x M)
    N, M,  # Dimensions
    stride_b_row, stride_b_col,  # Strides for B matrix
    stride_c_row, stride_c_col,  # Strides for C matrix
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID corresponds to the row index (i)
    row_idx = tl.program_id(0)
    
    # Load the diagonal element A[row_idx]
    a_val = tl.load(a_ptr + row_idx)
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    # Loop over columns in blocks
    for m_offset in range(0, M, BLOCK_SIZE_M):
        col_idx = m_offset + col_offsets
        # Create mask for valid column indices
        mask = col_idx < M
        
        # Load B[row_idx, col_idx]
        b_vals = tl.load(b_ptr + row_idx * stride_b_row + col_idx * stride_b_col, mask=mask)
        
        # Compute C[row_idx, col_idx] = A[row_idx] * B[row_idx, col_idx]
        c_vals = a_val * b_vals
        
        # Store result to C[row_idx, col_idx]
        tl.store(c_ptr + row_idx * stride_c_row + col_idx * stride_c_col, c_vals, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B using a custom Triton kernel.
    This is equivalent to element-wise multiplication of each row of B by the corresponding element in A.
    
    Args:
        A: 1D tensor of shape (N,) representing diagonal elements
        B: 2D tensor of shape (N, M)
    
    Returns:
        C: 2D tensor of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    
    # Prepare output tensor
    C = torch.empty_like(B)
    
    # Set block size for columns
    BLOCK_SIZE_M = 256
    
    # Grid: one program per row (N programs)
    grid = (N,)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for diag(A) @ B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)