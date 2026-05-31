import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal vector (1D tensor of size N)
    B_ptr,  # Pointer to matrix B (2D tensor of shape (N, M))
    C_ptr,  # Pointer to output matrix C (2D tensor of shape (N, M))
    N,      # Number of rows
    M,      # Number of columns
    stride_A,  # Stride for A (typically 1 for contiguous 1D tensor)
    stride_B,  # Stride for B (row stride for 2D tensor)
    stride_C,  # Stride for C (row stride for 2D tensor)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute row index for A and C
    row_idx = pid_m
    
    # Check if this row is in bounds
    row_mask = row_idx < N
    
    # Load diagonal element A[row_idx]
    a_val = tl.load(A_ptr + row_idx * stride_A, mask=row_mask)
    
    # Compute starting column indices for this block
    col_offsets = tl.arange(0, BLOCK_M)
    col_mask = col_offsets < M
    
    # Compute base pointer for row in B and C
    b_ptr = B_ptr + row_idx * stride_B + col_offsets
    c_ptr = C_ptr + row_idx * stride_C + col_offsets
    
    # Load B values for this row
    b_vals = tl.load(b_ptr, mask=col_mask, other=0.0)
    
    # Compute result: C[row_idx, :] = A[row_idx] * B[row_idx, :]
    c_vals = a_val * b_vals
    
    # Store result
    tl.store(c_ptr, c_vals, mask=col_mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B using a custom Triton kernel.
    
    Args:
        A: 1D tensor of shape (N,) representing the diagonal
        B: 2D tensor of shape (N, M)
    
    Returns:
        C: 2D tensor of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be a 1D tensor"
    assert B.dim() == 2, "B must be a 2D tensor"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    
    # Allocate output tensor
    C = torch.empty_like(B)
    
    # Set block sizes
    BLOCK_M = 128
    BLOCK_N = 1  # Since we only need one block for the N dimension (rows)
    
    # Grid: one block per row for N, and ceil(M/BLOCK_M) blocks for columns
    grid = (N, (M + BLOCK_M - 1) // BLOCK_M)
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0) if A.dim() > 1 else 1,
        B.stride(0),
        C.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    Uses a custom Triton kernel for optimization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)