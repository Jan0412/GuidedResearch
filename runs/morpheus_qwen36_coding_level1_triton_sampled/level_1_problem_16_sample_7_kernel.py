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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for C = A^T @ B where A is (K, M) and B is (K, N).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for M and N dimensions
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block
        # A is (K, M), so element (k, m) is at k * stride_ak + m * stride_am
        a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
        a_mask = (offs_k[:, None] < K) & mask_m[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block
        # B is (K, N), so element (k, n) is at k * stride_bk + n * stride_bn
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute dot product: acc += A_block^T @ B_block
        # a is (BLOCK_K, BLOCK_M), b is (BLOCK_K, BLOCK_N)
        acc += tl.dot(a.T, b)

    # Store result
    # C is (M, N), so element (m, n) is at m * stride_cm + n * stride_cn
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Grid configuration
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T @ B using Triton kernel.
        """
        return triton_matmul(A, B)


def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(K, M)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []