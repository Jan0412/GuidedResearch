import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A @ B.
    A shape: (M, K)
    B shape: (K, N)
    C shape: (M, N)
    """
    # Block IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the current block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks to handle boundaries
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_k = offs_k < K

    # Pointers to the first elements of the blocks
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks with masking
        a = tl.load(a_ptrs, mask=mask_am[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & mask_bn[None, :], other=0.0)

        # Matrix multiplication
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Inputs must be FP32."

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Block sizes (tunable)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid calculation
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom Triton kernel.
        Computes C = A.T @ B.T equivalent to matmul(A.T, B.T).
        """
        # Transpose inputs to match kernel expectation: A_t (M, K), B_t (K, N)
        A_t = A.T.contiguous()
        B_t = B.T.contiguous()

        # Call Triton kernel
        return triton_matmul(A_t, B_t)