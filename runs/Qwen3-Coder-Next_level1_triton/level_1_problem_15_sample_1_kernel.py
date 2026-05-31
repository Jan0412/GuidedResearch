import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for valid indices
    m_mask = offsets_m < N
    n_mask = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    num_k_tiles = tl.cdiv(N, BLOCK_K)
    for k in range(0, num_k_tiles):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offsets_k < N
        
        # For lower triangular matrices:
        # A is non-zero only when row >= col, so A[i,k] is non-zero when i >= k
        # B is non-zero only when row >= col, so B[k,j] is non-zero when k >= j
        
        # Load A: only load when row index >= column index (lower triangular)
        a_row_offsets = offsets_m[:, None]
        a_col_offsets = offsets_k[None, :]
        a_mask = m_mask[:, None] & k_mask[None, :] & (a_row_offsets >= a_col_offsets)
        a = tl.load(A_ptr + a_row_offsets * stride_am + a_col_offsets * stride_an, 
                   mask=a_mask, other=0.0)
        
        # Load B: only load when row index >= column index (lower triangular)
        b_row_offsets = offsets_k[:, None]
        b_col_offsets = offsets_n[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :] & (b_row_offsets >= b_col_offsets)
        b = tl.load(B_ptr + b_row_offsets * stride_bm + b_col_offsets * stride_bn, 
                   mask=b_mask, other=0.0)
        
        # Accumulate: only compute where row >= col in result
        c_row_offsets = offsets_m[:, None]
        c_col_offsets = offsets_n[None, :]
        c_mask = m_mask[:, None] & n_mask[None, :] & (c_row_offsets >= c_col_offsets)
        
        # Only accumulate where the result should be non-zero (lower triangular)
        acc += tl.where(c_mask, tl.dot(a, b, out_dtype=tl.float32), 0.0)
    
    # Store result
    c_row_offsets = offsets_m[:, None]
    c_col_offsets = offsets_n[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :] & (c_row_offsets >= c_col_offsets)
    
    tl.store(C_ptr + c_row_offsets * stride_cm + c_col_offsets * stride_cn,
             acc, mask=c_mask)


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
    C = torch.zeros_like(A)
    
    # Set block sizes for optimization
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B,
        returning only the lower triangular part of the result.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_tril_matmul(A, B)