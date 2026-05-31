import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Map program ID to the block of C it computes
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create offsets for the blocks
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (K, M), B is (N, K)
    # We want A[rk, rm] and B[rn, rk]
    a_ptr = A_ptr + (rk[:, None] * stride_ak + rm[None, :] * stride_am)
    b_ptr = B_ptr + (rn[None, :] * stride_bn + rk[:, None] * stride_bk)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        # a shape: (BLOCK_SIZE_K, BLOCK_SIZE_M)
        # b shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        a = tl.load(a_ptr, mask=(rk[:, None] + k * BLOCK_SIZE_K < K) & (rm[None, :] < M), other=0.0)
        b = tl.load(b_ptr, mask=(rn[None, :] < N) & (rk[:, None] + k * BLOCK_SIZE_K < K), other=0.0)

        # We need to perform: accumulator += trans(a) @ trans(b)
        # trans(a) is (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # trans(b) is (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator += tl.dot(tl.trans(a), tl.trans(b))

        # Advance pointers to the next block in K
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computes C = A.T @ B.T
    A shape: (K, M) -> A.T shape: (M, K)
    B shape: (N, K) -> B.T shape: (K, N)
    C shape: (M, N)
    """
    K, M = A.shape
    N, _ = B.shape
    
    # Ensure tensors are contiguous and on CUDA
    A = A.contiguous()
    B = B.contiguous()
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Strides
    stride_ak, stride_am = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tunable parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: one program per block of C
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_ak, stride_am,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A.T * B.T)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A.T @ B.T.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)