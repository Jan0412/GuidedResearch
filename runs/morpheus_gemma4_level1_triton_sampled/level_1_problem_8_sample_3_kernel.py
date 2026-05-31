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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A * B.
    A: (M, K), B: (K, N), C: (M, N)
    """
    # Map program ID to the block of C it computes
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create offsets for the current block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load block from A
        # a_ptr + (offs_am[:, None] * stride_am + (k * BLOCK_K + offs_k[None, :]) * stride_ak)
        a_offs = offs_am[:, None] * stride_am + (k * BLOCK_K + offs_k[None, :]) * stride_ak
        a_mask = (offs_am[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K)
        a_block = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        # Load block from B
        # b_ptr + ((k * BLOCK_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        b_offs = (k * BLOCK_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        b_mask = ((k * BLOCK_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
        b_block = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        # Matrix multiply and accumulate
        accumulator += tl.dot(a_block, b_block)

    # Store the result block in C
    c_offs = offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr + c_offs, accumulator, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the matmul_kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous for easier pointer arithmetic
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid definition
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Ensure inputs are FP32 as requested
        A = A.float()
        B = B.float()
        return triton_matmul(A, B)