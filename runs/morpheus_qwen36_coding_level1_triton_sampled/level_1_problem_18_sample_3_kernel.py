import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_n, stride_b_k,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    m_offsets = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    A_block_ptrs = A_ptr + m_offsets[:, None] * stride_a_m + k_offsets[None, :] * stride_a_k
    B_block_ptrs = B_ptr + k_offsets[:, None] * stride_b_k + n_offsets[None, :] * stride_b_n
    C_block_ptrs = C_ptr + m_offsets[:, None] * stride_c_m + n_offsets[None, :] * stride_c_n
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        A_block = tl.load(A_block_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B_block = tl.load(B_block_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(A_block, B_block)
        
        A_block_ptrs += BLOCK_K * stride_a_k
        B_block_ptrs += BLOCK_K * stride_b_k
        
    tl.store(C_block_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[0]
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_a_m=A.stride(1), stride_a_k=A.stride(0),
        stride_b_n=B.stride(1), stride_b_k=B.stride(0),
        stride_c_m=C.stride(1), stride_c_n=C.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)