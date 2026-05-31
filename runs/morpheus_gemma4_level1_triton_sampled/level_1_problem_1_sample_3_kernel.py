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

    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # A is (M, K), B is (K, N)
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # Masking is necessary if M, N, or K are not multiples of the block size
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication of the blocks
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block along the K dimension
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid of programs
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
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
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using a Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)