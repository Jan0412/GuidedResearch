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
    # Map program IDs to the block of C being computed
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the M and N dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (K, N)
    a_ptr += (pid_m * BLOCK_SIZE_M * stride_am)
    b_ptr += (pid_n * BLOCK_SIZE_N * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # a_block shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # b_block shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        # Matrix multiplication of the blocks
        accumulator += tl.dot(a, b)

    # Store the result block in C
    c_ptr += (pid_m * BLOCK_SIZE_M * stride_cm)
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets,
        accumulator,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid for the kernel
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of two symmetric matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        # Use the custom Triton matmul implementation for FP32 speedup
        return triton_matmul(A, B)