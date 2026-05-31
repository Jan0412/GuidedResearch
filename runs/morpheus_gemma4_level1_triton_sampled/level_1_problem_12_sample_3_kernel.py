import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,          # Pointer to diagonal vector A
    B_ptr,          # Pointer to matrix B
    Out_ptr,        # Pointer to output matrix Out
    N,              # Number of rows in B
    M,              # Number of columns in B
    stride_B_row,   # Stride between rows of B
    stride_B_col,   # Stride between columns of B
    stride_Out_row, # Stride between rows of Out
    stride_Out_col, # Stride between columns of Out
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (or part of one row)
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Calculate column offsets for this block
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < M
    
    # Load the diagonal element for the current row
    # A is a 1D tensor of shape (N,)
    a_val = tl.load(A_ptr + row)
    
    # Load the corresponding block from matrix B
    # B is shape (N, M)
    b_vals = tl.load(
        B_ptr + row * stride_B_row + col_offsets * stride_B_col, 
        mask=mask
    )
    
    # Perform the multiplication: Out[row, col] = A[row] * B[row, col]
    res = a_val * b_vals
    
    # Store the result in the output matrix
    tl.store(
        Out_ptr + row * stride_Out_row + col_offsets * stride_Out_col, 
        res, 
        mask=mask
    )

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton implementation of diag(A) @ B.
    C[i, j] = A[i] * B[i, j]
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous to simplify pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # Strides for B and Out
    stride_B_row = B.stride(0)
    stride_B_col = B.stride(1)
    stride_Out_row = out.stride(0)
    stride_Out_col = out.stride(1)
    
    # Tuning parameter for block size (columns processed per program)
    BLOCK_SIZE = 1024
    
    # Grid: (Rows of B, Columns of B / BLOCK_SIZE)
    grid = (N, (M + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        stride_B_row, stride_B_col,
        stride_Out_row, stride_Out_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B, implemented via a custom Triton kernel for speedup.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)