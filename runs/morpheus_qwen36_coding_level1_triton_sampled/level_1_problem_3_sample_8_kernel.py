import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + pid_b * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    num_k_tiles = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    for k in range(num_k_tiles):
        mask_k = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b = tl.load(b_ptrs, mask=mask_k.T, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = C_ptr + pid_b * stride_cb + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)
    
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    num_pid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (batch_size, num_pid_m, num_pid_n)
    
    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)