import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_an, stride_am, stride_ak,
    stride_b0, stride_b1,
    stride_cn, stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K)
    offs_bk = tl.arange(0, BLOCK_K)
    offs_bl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    a_ptrs = A_ptr + pid_n * stride_an + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = B_ptr + offs_bk[:, None] * stride_b0 + offs_bl[None, :] * stride_b1

    accumulator = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)

    num_k = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_bk[:, None] < K) & (offs_bl[None, :] < L), other=0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_b0

    c_ptrs = C_ptr + pid_n * stride_cn + offs_am[:, None] * stride_cm + offs_bl[None, :] * stride_ck
    tl.store(c_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bl[None, :] < L))

def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2
    
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_L = 64
    
    grid = (N, (M + BLOCK_M - 1) // BLOCK_M, (L + BLOCK_L - 1) // BLOCK_L)
    
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        M*K, K, 1,
        L, 1,
        M*L, L, 1,
        BLOCK_M, BLOCK_K, BLOCK_L,
        num_warps=4
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)