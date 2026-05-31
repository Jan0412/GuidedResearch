import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to 1D diagonal vector A of shape (N,)
    b_ptr,  # Pointer to 2D matrix B of shape (N, M)
    c_ptr,  # Pointer to output matrix C of shape (N, M)
    N, M,  # Dimensions
    stride_b_m, stride_b_n,  # Strides for B matrix
    stride_c_m, stride_c_n,  # Strides for C matrix
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Block index for M dimension
    pid_n = tl.program_id(1)  # Block index for N dimension
    
    # Create offsets for N dimension (rows)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Create offsets for M dimension (columns)
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Create masks for bounds checking
    n_mask = n_offsets < N
    m_mask = m_offsets < M
    
    # Load diagonal elements of A (only needed for the rows this block processes)
    a_vals = tl.load(a_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Load B matrix block: shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # B is accessed as B[n, m], so row-major stride is (n * stride_b_m + m * stride_b_n)
    b_block = tl.load(
        b_ptr + n_offsets[:, None] * stride_b_m + m_offsets[None, :] * stride_b_n,
        mask=(n_mask[:, None] & m_mask[None, :]),
        other=0.0
    )
    
    # Compute C = diag(A) * B element-wise: C[i,j] = A[i] * B[i,j]
    # This is a simple scaling of each row of B by the corresponding diagonal element
    c_block = a_vals[:, None] * b_block
    
    # Store result
    tl.store(
        c_ptr + n_offsets[:, None] * stride_c_m + m_offsets[None, :] * stride_c_n,
        c_block,
        mask=(n_mask[:, None] & m_mask[None, :])
    )


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = diag(A) @ B using a custom Triton kernel.
    
    Args:
        A (torch.Tensor): 1D tensor of shape (N,)
        B (torch.Tensor): 2D tensor of shape (N, M)
    
    Returns:
        torch.Tensor: Result of shape (N, M)
    """
    assert A.dim() == 1, "A must be a 1D tensor"
    assert B.dim() == 2, "B must be a 2D tensor"
    assert A.size(0) == B.size(0), f"A and B must have same first dimension, got {A.size(0)} and {B.size(0)}"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    
    # Allocate output tensor
    C = torch.empty_like(B)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),  # Number of blocks in M dimension
        triton.cdiv(N, BLOCK_SIZE_N),  # Number of blocks in N dimension
    )
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)