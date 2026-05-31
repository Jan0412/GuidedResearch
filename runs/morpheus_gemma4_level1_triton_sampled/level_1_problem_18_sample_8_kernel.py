import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for C = A^T * B^T
    C[i, j] = sum_{k=0}^{K-1} A[k, i] * B[j, k]
    
    A is (K, M), B is (N, K), C is (M, N)
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Ranges for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension K
    for k in range(0, K, BLOCK_K):
        # A is (K, M). We want a block of A^T starting at (rm, k).
        # A^T[rm, k] = A[k, rm].
        # Offset in A: (k + rk) * stride_ak + rm * stride_am
        a_offsets = (k + rk)[:, None] * stride_ak + rm[None, :] * stride_am
        a_mask = (k + rk)[:, None] < K
        a_mask &= (rm[None, :] < M)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        # B is (N, K). We want a block of B^T starting at (k, rn).
        # B^T[k, rn] = B[rn, k].
        # Offset in B: rn * stride_bn + (k + rk) * stride_bk
        b_offsets = rn[:, None] * stride_bn + (k + rk)[None, :] * stride_bk
        b_mask = (rn[:, None] < N)
        b_mask &= (k + rk)[None, :] < K
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        # a is (BLOCK_K, BLOCK_M), b is (BLOCK_N, BLOCK_K)
        # We need tl.dot(A_block_T, B_block_T)
        # A_block_T = tl.trans(a) -> (BLOCK_M, BLOCK_K)
        # B_block_T = tl.trans(b) -> (BLOCK_K, BLOCK_N)
        acc += tl.dot(tl.trans(a), tl.trans(b))

    # Store the result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    A: (K, M)
    B: (N, K)
    Returns C: (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    N, K_check = B.shape
    assert K == K_check, "Inner dimensions must match."

    # Output tensor C: (M, N)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides
    # A is (K, M) row-major: stride_ak = M, stride_am = 1
    # B is (N, K) row-major: stride_bn = K, stride_bk = 1
    # C is (M, N) row-major: stride_cm = N, stride_cn = 1
    stride_ak, stride_am = A.stride() if A.shape == (K, M) else (M, 1) # Standard row-major
    # Since A is (K, M) and we treat it as A^T (M, K), we pass strides based on A's layout
    # A[k, i] offset = k * M + i
    s_ak = A.stride(0)
    s_am = A.stride(1)
    
    s_bn = B.stride(0)
    s_bk = B.stride(1)
    
    s_cm = C.stride(0)
    s_cn = C.stride(1)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        s_am, s_ak,
        s_bn, s_bk,
        s_cm, s_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A^T * B^T)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)