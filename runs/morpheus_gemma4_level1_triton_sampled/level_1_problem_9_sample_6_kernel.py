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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to A and B blocks
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator for the result block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiplication of tiles
        acc += tl.dot(a, b)
        
        # Advance pointers to the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result block to C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(a, b):
    # Ensure inputs are on GPU and contiguous for optimal performance
    a = a.cuda().contiguous() if not a.is_cuda else a.contiguous()
    b = b.cuda().contiguous() if not b.is_cuda else b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Tuning parameters: K=32 is small, so BLOCK_SIZE_K=32 processes the inner dim in one step
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
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel optimized for cases where the inner dimension K is small.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return triton_matmul(A, B)