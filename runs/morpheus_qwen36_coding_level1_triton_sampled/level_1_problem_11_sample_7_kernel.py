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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_ak = tl.arange(0, BLOCK_K)
    mask_am = offsets_am < M
    mask_ak = offsets_ak < K
    
    offsets_bk = tl.arange(0, BLOCK_K)
    offsets_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bk = offsets_bk < K
    mask_bn = offsets_bn < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offsets_am[:, None] * stride_am + (k + offsets_ak)[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_am[:, None] & (k + offsets_ak)[None, :] < K, other=0.0)
        
        b_ptrs = B_ptr + (k + offsets_bk)[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(k + offsets_bk)[:, None] < K & mask_bn[None, :], other=0.0)
        
        acc += tl.dot(a, b)
        
    offsets_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_cm = offsets_cm < M
    mask_cn = offsets_cn < N
    
    c_ptrs = C_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_cm[:, None] & mask_cn[None, :])


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    k = B.shape[1]
    N = b * i * j
    
    A_flat = A.reshape(N, l)
    C_flat = torch.empty(N, k, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(k, BLOCK_N))
    
    matmul_kernel[grid](
        A_flat, B, C_flat,
        N, k, l,
        stride_am=l, stride_ak=1,
        stride_bk=k, stride_bn=1,
        stride_cm=k, stride_cn=1,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return C_flat.reshape(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)