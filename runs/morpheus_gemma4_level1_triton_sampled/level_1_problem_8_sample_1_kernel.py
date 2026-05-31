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
    # Map program IDs to the blocks of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # a_ptr is shifted by row offset and column offset
    # b_ptr is shifted by row offset and column offset
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B with boundary masking
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication of the blocks
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers to the next block along the K dimension
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Compute the pointer for the output block in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    
    # Store the final result in C with boundary masking
    tl.store(c_ptr, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are on CUDA and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "Incompatible shapes for matrix multiplication."

    # Prepare output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
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
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a Triton kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return triton_matmul(A, B)