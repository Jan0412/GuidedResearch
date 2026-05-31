import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_tri_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for multiplying two lower triangular matrices A and B.
    C = A * B. 
    Since A and B are lower triangular, C is also lower triangular.
    C[i, j] = sum_{k=j}^i A[i, k] * B[k, j] for i >= j, and 0 otherwise.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block boundaries
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = m_start + BLOCK_SIZE_M
    n_end = n_start + BLOCK_SIZE_N

    # Optimization: If the entire block is in the upper triangle (i < j), skip it.
    # i < m_end and j >= n_start. If m_end <= n_start, then i < j for all i, j in block.
    if m_end <= n_start:
        return

    # Offsets for the blocks
    rm = m_start + tl.arange(0, BLOCK_SIZE_M)
    rn = n_start + tl.arange(0, BLOCK_SIZE_N)

    # Pointers to the start of the blocks in A and B
    # A is (N, N), B is (N, N)
    # A_ptr + rm * stride_am + rk * stride_ak
    # B_ptr + rk * stride_bk + rn * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # The k-loop: for C[i, j], we only need k in [j, i].
    # For the block (pid_m, pid_n), the relevant k range is [n_start, m_end).
    # We iterate in steps of BLOCK_SIZE_K.
    k_start = n_start
    k_end = m_end
    
    # Ensure k_start and k_end are aligned to BLOCK_SIZE_K for simplified indexing
    # though we will use masks for boundaries.
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Mask for k to stay within [0, N)
        k_mask = rk < N
        
        # Load A block: A[rm, rk]
        # Mask: rm < N and rk < N. 
        # Additionally, A is lower triangular, so A[i, k] = 0 if k > i.
        # The tl.load will handle the boundary, and the 0s are inherent in the data.
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
                    mask=(rm[:, None] < N) & (rk[None, :] < N), other=0.0)
        
        # Load B block: B[rk, rn]
        # Mask: rk < N and rn < N.
        # B is lower triangular, so B[k, j] = 0 if k < j.
        b = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, 
                    mask=(rk[:, None] < N) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiply-accumulate
        accumulator += tl.dot(a, b)

    # Final mask to ensure we only store the lower triangular part (i >= j)
    # and stay within the matrix bounds.
    out_mask = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] >= rn[None, :])
    
    # Store the result
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, 
             accumulator, mask=out_mask, other=0.0)


def triton_matmul_tri(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    N = A.shape[0]
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    C = torch.empty((N, N), device=A.device, dtype=torch.float32)
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_tri_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    Utilizes a custom Triton kernel to exploit the triangular structure for speedup.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B using Triton.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # The inputs are expected to be lower triangular.
        # The custom kernel computes the product and ensures the result is lower triangular.
        return triton_matmul_tri(A, B)