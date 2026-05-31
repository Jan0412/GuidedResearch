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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to the block of C being computed
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first elements of the blocks in A and B
    # A is (M, K), B is (K, N)
    # a_ptr is base of A, b_ptr is base of B
    
    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Offset for the current K-block
        rk_offset = k * BLOCK_SIZE_K + rk
        
        # Load blocks from A and B
        # A block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # B block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk_offset[None, :] * stride_ak),
            mask=(rm[:, None] < M) & (rk_offset[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk_offset[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=(rk_offset[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        
        # Perform matrix multiplication of blocks
        accumulator += tl.dot(a, b)

    # Store the result back to C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets, 
        accumulator, 
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Shape information
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Tuning parameters
    # Given K=64, we can set BLOCK_SIZE_K=64 to process the inner dimension in one go.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # Grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
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
        Performs matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are contiguous for the kernel
        A = A.contiguous()
        B = B.contiguous()
        return triton_matmul(A, B)