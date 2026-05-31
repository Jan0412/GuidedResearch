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
    # Map program IDs to the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to the start of the blocks in A and B
    # a_ptr is (M, K), b_ptr is (K, N)
    a_ptr += pid_m * stride_am
    b_ptr += pid_n * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks of A and B
        # A block: [BLOCK_M, BLOCK_K], B block: [BLOCK_K, BLOCK_N]
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + (k * BLOCK_K + rk[None, :]) * stride_ak),
            mask=(rm[:, None] < M) & ((k * BLOCK_K + rk[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((k * BLOCK_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn),
            mask=((k * BLOCK_K + rk[:, None]) < K) & (rn[None, :] < N),
            other=0.0,
        )
        # Dot product and accumulate
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_ptr += pid_m * stride_cm
    tl.store(
        c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        accumulator,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on CUDA
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

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
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel for speedups on large K dimensions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)