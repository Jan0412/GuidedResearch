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
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    
    offs_am = pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_am = offs_am < N
    mask_bn = offs_bn < N
    
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask_am[:, None] & (offs_k[None, :] < N - k), other=0.0)
        b = tl.load(b_ptrs, mask=mask_bn[None, :] & (offs_k[:, None] < N - k), other=0.0)
        
        accumulator = tl.dot(a, b, accumulator=accumulator)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    offs_cm = pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_c = mask_am[:, None] & mask_bn[None, :]
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.empty_like(A)
    
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    grid = ((N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    matmul_kernel[grid](
        A, B, C,
        N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)