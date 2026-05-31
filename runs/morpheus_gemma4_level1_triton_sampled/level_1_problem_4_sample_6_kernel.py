import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    a_ptr,  # Pointer to matrix A
    b_ptr,  # Pointer to vector B
    out_ptr,  # Pointer to output vector C
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Row offsets for this block
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Mask for rows to handle cases where M is not a multiple of BLOCK_SIZE_M
    mask_m = rm < M
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        # Mask for columns to handle cases where K is not a multiple of BLOCK_SIZE_K
        mask_k = rk < K
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_vals = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
            mask=mask_m[:, None] & mask_k[None, :], 
            other=0.0
        )
        
        # Load B block: shape (BLOCK_SIZE_K,)
        # B is (K, 1), so we treat it as a 1D vector
        b_vals = tl.load(
            b_ptr + rk * stride_bk, 
            mask=mask_k, 
            other=0.0
        )
        
        # Compute partial dot product: (BLOCK_SIZE_M, BLOCK_SIZE_K) * (BLOCK_SIZE_K,)
        # Result is (BLOCK_SIZE_M,)
        acc += tl.sum(a_vals * b_vals[None, :], axis=1)
        
    # Store the final result back to memory
    tl.store(out_ptr + rm, acc, mask=mask_m)


def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for matrix-vector multiplication (C = A * B).
    A: (M, K), B: (K, 1) -> C: (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    
    # Tuning parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 1024
    
    # Grid: Parallelize over the M dimension
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,)
    
    matvec_kernel[grid](
        A, B, out,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)