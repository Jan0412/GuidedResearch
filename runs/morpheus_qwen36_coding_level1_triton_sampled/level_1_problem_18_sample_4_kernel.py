import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    A_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    B_offsets = k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        A_tile = tl.load(A_ptr + A_offsets, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B_tile = tl.load(B_ptr + B_offsets, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(A_tile, B_tile, out_dtype=tl.float32)
        A_offsets += BLOCK_K * stride_ak
        B_offsets += BLOCK_K * stride_bk
        
    C_offsets = m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptr + C_offsets, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A_t = A.T.contiguous()
    B_t = B.T.contiguous()
    
    M, K_dim = A_t.shape
    _, N = B_t.shape
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = lambda meta: ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    
    matmul_kernel[grid](A_t, B_t, C, M, N, K_dim, 
                        A_t.stride(0), A_t.stride(1), 
                        B_t.stride(0), B_t.stride(1), 
                        C.stride(0), C.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


def get_inputs():
    A = torch.rand(4096 * 2, 1024 * 2)
    B = torch.rand(2048 * 2, 4096 * 2)
    return [A, B]


def get_init_inputs():
    return []