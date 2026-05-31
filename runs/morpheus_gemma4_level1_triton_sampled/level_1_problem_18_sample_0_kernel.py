import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to matrix A (K, M)
    b_ptr,  # Pointer to matrix B (N, K)
    c_ptr,  # Pointer to output matrix C (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel to compute C = A.T @ B.T
    C[i, j] = sum_k (A.T)[i, k] * (B.T)[k, j] = sum_k A[k, i] * B[j, k]
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

    # Loop over the K dimension
    for k in range(0, K, BLOCK_K):
        k_offset = k + rk
        
        # Load block from A: A[k_offset, rm]
        # A is (K, M). Index is k * stride_am + i * stride_ak
        # We want a block of shape (BLOCK_M, BLOCK_K) where element (i, k) is A[k, i]
        a_ptr_block = a_ptr + (k_offset[None, :] * stride_am + rm[:, None] * stride_ak)
        a_mask = (rm[:, None] < M) & (k_offset[None, :] < K)
        a = tl.load(a_ptr_block, mask=a_mask, other=0.0)

        # Load block from B: B[rn, k_offset]
        # B is (N, K). Index is j * stride_bn + k * stride_bk
        # We want a block of shape (BLOCK_K, BLOCK_N) where element (k, j) is B[j, k]
        b_ptr_block = b_ptr + (rn[None, :] * stride_bn + k_offset[:, None] * stride_bk)
        b_mask = (rn[None, :] < N) & (k_offset[:, None] < K)
        b = tl.load(b_ptr_block, mask=b_mask, other=0.0)

        # Compute dot product: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, b)

    # Store the result in C
    c_ptr_block = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr_block, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    A: (K, M)
    B: (N, K)
    Returns: C = A.T @ B.T (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()

    K, M = A.shape
    N, K_check = B.shape
    assert K == K_check, "Inner dimensions must match"

    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides
    stride_am, stride_ak = M, 1 # A is (K, M)
    stride_bn, stride_bk = K, 1 # B is (N, K)
    stride_cm, stride_cn = N, 1 # C is (M, N)

    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs C = A.T * B.T using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B.T.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)