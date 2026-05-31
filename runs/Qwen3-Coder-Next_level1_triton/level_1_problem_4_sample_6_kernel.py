import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_bw,
    stride_cm, stride_cw,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    
    # Create m offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load A block: (BLOCK_M, BLOCK_K)
        a_offsets = (
            m_offsets[:, None] * stride_am + 
            k_offsets[None, :] * stride_ak
        )
        a = tl.load(A_ptr + a_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B vector: (BLOCK_K, 1)
        b_offsets = k_offsets[:, None] * stride_bk + tl.arange(0, 1)[None, :] * stride_bw
        b = tl.load(B_ptr + b_offsets, mask=k_mask[:, None], other=0.0)
        
        # Matrix multiplication accumulation
        accumulator += tl.dot(a, b, allow_tf32=False)
    
    # Store result
    c_offsets = (
        m_offsets[:, None] * stride_cm + 
        tl.arange(0, 1)[None, :] * stride_cw
    )
    tl.store(C_ptr + c_offsets, accumulator, mask=m_mask[:, None])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input vector of shape (K, 1)
        
    Returns:
        Output vector of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    assert B.shape == (K, 1), f"B must be of shape ({K}, 1), got {B.shape}"
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_K = 256
    
    # Grid: one block per BLOCK_M rows
    grid = (triton.cdiv(M, BLOCK_M),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)