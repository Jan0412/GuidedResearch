import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for C = A^T * B
    A is (K, M), B is (K, N), C is (M, N)
    """
    # Map program IDs to the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for the first blocks of A and B
    # For A^T, the "row" index is M and "col" index is K.
    # A^T[m, k] = A[k, m]. 
    # So stride_am (row stride for A^T) is 1, and stride_ak (col stride for A^T) is M.
    a_ptr += rm[None, :] * stride_am + rk[:, None] * stride_ak
    b_ptr += rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        # A block is (BLOCK_SIZE_K, BLOCK_SIZE_M), B block is (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We need A^T block (BLOCK_SIZE_M, BLOCK_SIZE_K) and B block (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Note: we load A as (K, M) and then transpose it in the dot product or load it such that 
        # it's compatible. In Triton, tl.dot expects (M, K) and (K, N).
        
        # Current slice of A: A[k*BK : (k+1)*BK, rm] -> shape (BK, BM)
        # We want A^T[rm, k*BK : (k+1)*BK] -> shape (BM, BK)
        a = tl.load(a_ptr, mask=(rk[:, None] < (K - k * BLOCK_SIZE_K)) & (rm[None, :] < M), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < (K - k * BLOCK_SIZE_K)) & (rn[None, :] < N), other=0.0)

        # Since 'a' is (BK, BM), we transpose it to (BM, BK) for tl.dot
        accumulator += tl.dot(tl.trans(a), b)

        # Advance pointers to the next block in K
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr += rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel to compute A.T @ B
    A: (K, M)
    B: (K, N)
    Returns: (M, N)
    """
    K, M = A.shape
    K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match"

    # Ensure tensors are contiguous and on CUDA
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides
    # A is (K, M) -> A.T is (M, K). A.T[i, j] = A[j, i]
    # stride_am: distance between A.T[i, j] and A.T[i+1, j] = A[j, i] and A[j, i+1] = 1
    # stride_ak: distance between A.T[i, j] and A.T[i, j+1] = A[j, i] and A[j+1, i] = M
    stride_am = 1
    stride_ak = M
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A^T * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A.T @ B.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)