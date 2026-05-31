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
    # Map program IDs to the block of C it computes
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    rm = pid % num_pid_m
    rn = pid // num_pid_m

    # Pointers to the start of the blocks in A and B
    a_offset = rm * BLOCK_SIZE_M * stride_am
    b_offset = rn * BLOCK_SIZE_N * stride_bn

    # Create ranges for the blocks
    rm_range = tl.arange(0, BLOCK_SIZE_M)
    rn_range = tl.arange(0, BLOCK_SIZE_N)
    rk_range = tl.arange(0, BLOCK_SIZE_K)

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load block from A
        a_ptrs = a_ptr + a_offset + (rm_range[:, None] * stride_am + (k * BLOCK_SIZE_K + rk_range[None, :]) * stride_ak)
        a_mask = (rm_range[:, None] < M) & ((k * BLOCK_SIZE_K + rk_range[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load block from B
        b_ptrs = b_ptr + b_offset + ((k * BLOCK_SIZE_K + rk_range[:, None]) * stride_bk + rn_range[None, :] * stride_bn)
        b_mask = ((k * BLOCK_SIZE_K + rk_range[:, None]) < K) & (rn_range[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiplication of blocks
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_ptrs = c_ptr + (rm_range[:, None] * stride_cm + rn_range[None, :] * stride_cn)
    c_mask = (rm_range[:, None] < M) & (rn_range[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are on CUDA and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a custom Triton kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Ensure inputs are FP32 for the kernel
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return triton_matmul(A, B)