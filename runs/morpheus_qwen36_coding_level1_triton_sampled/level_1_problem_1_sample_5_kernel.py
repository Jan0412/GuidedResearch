import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_ck,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    k_offsets = tl.arange(0, TILE_K)
    
    a_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    b_offsets = k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    
    a_mask = (m_offsets[:, None] < N) & (k_offsets[None, :] < N)
    b_mask = (k_offsets[:, None] < N) & (n_offsets[None, :] < N)
    
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    for k in range(0, N, TILE_K):
        a = tl.load(A_ptr + a_offsets + k * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + b_offsets + k * stride_bk, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        
    c_offsets = m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_ck
    c_mask = (m_offsets[:, None] < N) & (n_offsets[None, :] < N)
    
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    out = torch.empty((N, N), dtype=torch.float32, device=A.device)
    
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    
    grid = lambda meta: (
        (N + meta["TILE_M"] - 1) // meta["TILE_M"],
        (N + meta["TILE_N"] - 1) // meta["TILE_N"],
        1
    )
    
    matmul_kernel[grid](
        A, B, out, N,
        N, 1,
        N, 1,
        N, 1,
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K
    )
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []