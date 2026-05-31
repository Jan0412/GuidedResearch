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
    Optimized for cases where K is small (tall/skinny matrices).
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Ranges for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for the result block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the inner dimension K in blocks of BLOCK_K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Compute pointers for the current blocks of A and B
        a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        # Masks to handle boundary conditions
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        
        # Load blocks from global memory
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform matrix multiplication of blocks
        acc += tl.dot(a, b)

    # Compute pointers for the output block and store the result
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    # Ensure inputs are on CUDA and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Hyperparameters for tiling
    # Since K is small (e.g., 32), we set BLOCK_K to 32
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

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
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using the Triton-optimized kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return triton_matmul(A, B)