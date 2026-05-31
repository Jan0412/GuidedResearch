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
    # Block indices
    block_m_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for M and N dimensions
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Accumulator initialized to zeros in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load block A
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        mask_a = mask_m[:, None] & (k + offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load block B
        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        mask_b = mask_n[None, :] & (k + offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Matrix multiply and accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication optimized for large K dimensions.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, "Inner dimensions must match."
    
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Tunable block sizes
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K1,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)