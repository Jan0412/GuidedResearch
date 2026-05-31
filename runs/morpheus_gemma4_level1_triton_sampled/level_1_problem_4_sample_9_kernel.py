import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for the M dimension
    pid_m = tl.program_id(0)
    
    # Row offsets for this block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = rm < M
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Column offsets for this block
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = rk < K
        
        # Load a block of A (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
            mask=mask_m[:, None] & mask_k[None, :], 
            other=0.0
        )
        
        # Load a block of B (BLOCK_SIZE_K,)
        b = tl.load(
            B_ptr + rk * stride_bk, 
            mask=mask_k, 
            other=0.0
        )
        
        # Perform partial dot product and accumulate
        # a: (BLOCK_SIZE_M, BLOCK_SIZE_K), b: (BLOCK_SIZE_K,)
        # Result of a * b[None, :] is (BLOCK_SIZE_M, BLOCK_SIZE_K)
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store the final result in C (M, 1)
    tl.store(C_ptr + rm * stride_cm, acc, mask=mask_m)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match"
    assert N == 1, "B must be a vector (K, 1)"
    
    # Output tensor
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk = B.stride(0)
    stride_cm = C.stride(0)
    
    # Tuning parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 1024
    
    # Grid: parallelize over rows of A
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    
    matmul_kernel[grid](
        A, B, C,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        stride_cm,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using a custom Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)