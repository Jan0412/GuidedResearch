import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Dimension of the square matrices
    stride_a,  # Stride for A matrix (typically N)
    stride_b,  # Stride for B matrix (typically N)
    stride_c,  # Stride for C matrix (typically N)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of the output matrix
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Only compute for lower triangular part (row >= col)
    if row_idx >= col_idx:
        # Compute the dot product for C[row_idx, col_idx]
        acc = 0.0
        # For lower triangular matrices, we need to sum over k where A[row_idx, k] and B[k, col_idx] are non-zero
        # Since A is lower triangular: A[row_idx, k] = 0 if k > row_idx
        # Since B is lower triangular: B[k, col_idx] = 0 if k < col_idx
        # So the valid range is k from col_idx to row_idx
        for k in range(col_idx, row_idx + 1):
            a_val = tl.load(A_ptr + row_idx * stride_a + k)
            b_val = tl.load(B_ptr + k * stride_b + col_idx)
            acc += a_val * b_val
        
        # Store the result
        tl.store(C_ptr + row_idx * stride_c + col_idx, acc)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of lower triangular matrices A and B,
    returning only the lower triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices A and B must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Grid: each block handles one element of the lower triangular part
    # We only need to compute the lower triangular part, so we use a grid that covers N x N
    # but the kernel will only compute for row >= col
    grid = (N, N)
    
    # Launch the Triton kernel
    tril_matmul_kernel[grid](
        A, B, C, 
        N,
        stride_a=N,
        stride_b=N,
        stride_c=N,
        BLOCK_SIZE=1  # Since we're computing one element per program
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_tril_matmul(A, B)