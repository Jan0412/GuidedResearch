import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID mapping
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for M and N dimensions
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Masks for M and N bounds
    mask_m = m_offsets < M
    mask_n = n_offsets < N
    
    # Base pointers for A and B tiles
    # A shape (M, K), strides (K, 1)
    # B shape (K, N), strides (N, 1)
    a_base = A_ptr + m_offsets[:, None] * stride_am
    b_base = B_ptr + n_offsets[None, :] * stride_bn
    
    # Accumulator for the result tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Global K indices for current block
        k_idx = k + k_offsets
        
        # Mask for K bounds
        mask_k = k_idx < K
        
        # Compute masks for A and B tiles
        # A tile mask: (BLOCK_M, BLOCK_K)
        a_mask = mask_m[:, None] & mask_k[None, :]
        # B tile mask: (BLOCK_K, BLOCK_N)
        b_mask = mask_k[:, None] & mask_n[None, :]
        
        # Load A tile
        a_ptrs = a_base + k_idx[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile
        b_ptrs = b_base + k_idx[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform dot product and accumulate
        acc += tl.dot(a, b)
        
    # Store the result tile to C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A, B):
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype == torch.float32, "Inputs must be FP32."
    assert B.dtype == torch.float32, "Inputs must be FP32."
    assert A.shape[1] == B.shape[0], "Inner dimensions must match."
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom Triton kernel.
        """
        return triton_matmul(A, B)