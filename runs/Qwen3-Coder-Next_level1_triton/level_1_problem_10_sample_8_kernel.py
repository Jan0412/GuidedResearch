import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to input 3D tensor A of shape (N, M, K)
    B_ptr,  # Pointer to input matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor C of shape (N, M, L)
    N, M, K, L,
    stride_an, stride_am, stride_ak,  # Strides for A
    stride_bk, stride_bl,            # Strides for B
    stride_cn, stride_cm, stride_cl,  # Strides for C
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Batch index (n_idx) and row index (m_idx) for the 3D tensor A
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    # Initialize output accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load block of A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A is indexed as [n_idx, m_idx, k_offsets]
        a_offsets = (
            n_idx * stride_an + 
            m_idx * stride_am + 
            k_offsets * stride_ak
        )
        a_block = tl.load(A_ptr + a_offsets, mask=k_mask, other=0.0)
        
        # Load block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        # B is indexed as [k_offsets, l_offsets]
        l_offsets = tl.arange(0, BLOCK_SIZE_L)
        l_mask = l_offsets < L
        b_offsets = (
            k_offsets[:, None] * stride_bk + 
            l_offsets[None, :] * stride_bl
        )
        b_block = tl.load(B_ptr + b_offsets, mask=k_mask[:, None] & l_mask[None, :], other=0.0)
        
        # Accumulate: a_block (M, K) @ b_block (K, L) = (M, L)
        accumulator += tl.dot(a_block, b_block)
    
    # Store result
    c_offsets = (
        n_idx * stride_cn + 
        m_idx * stride_cm + 
        tl.arange(0, BLOCK_SIZE_L) * stride_cl
    )
    c_mask = tl.arange(0, BLOCK_SIZE_L) < L
    tl.store(C_ptr + c_offsets, accumulator.to(tl.float32), mask=c_mask)


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton.
    
    Args:
        A: Input 3D tensor of shape (N, M, K)
        B: Input matrix of shape (K, L)
    
    Returns:
        Output tensor of shape (N, M, L)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, f"Dimension mismatch: A has K={K}, B has K={K_b}"
    
    # Prepare output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Compute strides
    stride_an, stride_am, stride_ak = A.stride()
    stride_bk, stride_bl = B.stride()
    stride_cn, stride_cm, stride_cl = C.stride()
    
    # Grid: (N, M) for the batched operation
    grid = (N, M)
    
    # Tunable block sizes for FP32
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 64
    
    # Launch kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_an, stride_am, stride_ak,
        stride_bk, stride_bl,
        stride_cn, stride_cm, stride_cl,
        BLOCK_SIZE_N=1,  # Not used in this grid but kept for interface
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul_3d(A, B)