import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Dimension of the matrices (N x N)
    stride_A, stride_B, stride_C,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element C[i,j] of the output matrix
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    # Only compute for upper triangular elements (i <= j)
    if i <= j:
        # Accumulator for the dot product
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Compute dot product of row i of A and column j of B
        # For upper triangular matrices:
        # A[i,k] is non-zero only when k >= i
        # B[k,j] is non-zero only when k <= j
        # So we only need to iterate k from i to j
        for k in range(i, j + 1):
            a_val = tl.load(A_ptr + i * stride_A + k)
            b_val = tl.load(B_ptr + k * stride_B + j)
            acc += a_val * b_val
        
        # Store the result
        tl.store(C_ptr + i * stride_C + j, acc)
    # Elements where i > j remain zero (upper triangular)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = triu(A @ B) for upper triangular matrices A and B
    using a custom Triton kernel that only computes upper triangular elements.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Strides for contiguous tensors
    stride_A = A.stride(0)
    stride_B = B.stride(0)
    stride_C = C.stride(0)
    
    # Grid: one block per output element (N x N grid)
    grid = (N, N)
    
    # Tune block size (though we're using 2D grid, so BLOCK_SIZE is not used here)
    # We use a simple 2D grid where each program computes one element
    triu_matmul_kernel[grid](
        A, B, C, 
        N,
        stride_A, stride_B, stride_C,
        BLOCK_SIZE=1,  # Not used in this implementation
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel that only computes upper triangular elements.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)