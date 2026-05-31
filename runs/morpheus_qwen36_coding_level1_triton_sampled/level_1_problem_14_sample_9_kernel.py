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
    m_block_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    
    m_offsets = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    k_mask = k_offsets < K
    
    a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k_blocks):
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c = accumulator
    
    mask = m_mask[:, None] & n_mask[None, :]
    upper_mask = m_offsets[:, None] <= n_offsets[None, :]
    mask = mask & upper_mask
    
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul_upper_tri(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    
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
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_upper_tri(A, B)