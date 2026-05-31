import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (N, K)
    # We want C = A @ B.T, so C[i, j] = sum_k A[i, k] * B[j, k]
    a_ptr += pid_m * BLOCK_SIZE_M * stride_am
    b_ptr += pid_n * BLOCK_SIZE_N * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load block from A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A[rm, rk]
        a_block = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        # Load block from B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We want B[rn, rk], but tl.dot expects (K, N)
        # So we load it such that the result is (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # B_block[k, n] = B[rn[n], rk[k]]
        b_block = tl.load(
            b_ptr + (rn[None, :] * stride_bn + rk[:, None] * stride_bk),
            mask=(rn[None, :] < N) & (rk[:, None] < K),
            other=0.0,
        )
        # Matrix multiply and accumulate
        accumulator += tl.dot(a_block, b_block)

    # Store the result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets,
        accumulator,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, "Inner dimensions must match."

    # Output tensor
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = out.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B^T) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A @ B.T.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)