import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_transpose_a_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block of C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load block from A. A is shape (K, M).
        # We want A[rk, rm], so we use stride_ak for the K-dimension and stride_am for the M-dimension.
        a_ptr = A_ptr + rk[:, None] * stride_ak + rm[None, :] * stride_am
        a = tl.load(a_ptr, mask=(rk[:, None] < K) & (rm[None, :] < M), other=0.0)

        # Load block from B. B is shape (K, N).
        # We want B[rk, rn], so we use stride_bk for the K-dimension and stride_bn for the N-dimension.
        b_ptr = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # The operation is C = A^T * B. 
        # a is (BLOCK_SIZE_K, BLOCK_SIZE_M). 
        # To perform the matrix multiplication, we transpose 'a' to (BLOCK_SIZE_M, BLOCK_SIZE_K).
        # (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) -> (BLOCK_SIZE_M, BLOCK_SIZE_N).
        acc += tl.dot(tl.trans(a), b)

    # Store the result in C
    c_ptr = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul_transpose_a(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton kernel that computes C = A^T * B.
    A: (K, M)
    B: (K, N)
    Output: (M, N)
    """
    # Ensure inputs are on CUDA and contiguous for predictable strides
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    K_B, N = B.shape
    assert K == K_B, "K dimensions of A and B must match"

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Block sizes for tiling
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions: one program for each block of the output C (M, N)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul_transpose_a_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(0), # stride_am, stride_ak
        B.stride(1), B.stride(0), # stride_bn, stride_bk
        C.stride(0), C.stride(1), # stride_cm, stride_cn
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A^T * B)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A^T * B.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose_a(A, B)