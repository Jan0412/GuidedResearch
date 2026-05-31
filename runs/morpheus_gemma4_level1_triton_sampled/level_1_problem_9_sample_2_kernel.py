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

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (K, N)
    a_ptr += pid_m * stride_am
    b_ptr += pid_n * stride_bn

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    # Since K=32 in this specific case, this loop will likely run once if BLOCK_SIZE_K=32
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        # a_block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # b_block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk[None, :]) * stride_ak),
            mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((k * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn),
            mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N),
            other=0.0,
        )
        # Matrix multiplication of blocks
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_ptr += pid_n * stride_cn
    tl.store(
        c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        accumulator,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

def triton_matmul(a, b):
    # Ensure inputs are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions"

    # Ensure tensors are contiguous for the kernel
    a = a.contiguous()
    b = b.contiguous()
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32 # Optimized for the given K=32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

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
    Optimized model that performs matrix multiplication (C = A * B) using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Use the custom Triton matmul implementation
        return triton_matmul(A, B)