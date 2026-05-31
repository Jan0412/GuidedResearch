import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_upper_tri_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k_a = offs_k[None, :] < K - k * BLOCK_K
        mask_k_b = offs_k[:, None] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_k_b, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_upper = row_idx[:, None] <= col_idx[None, :]
    mask_lower = ~mask_upper
    
    c_ptrs = C_ptr + row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_upper)
    tl.store(c_ptrs, 0.0, mask=mask_lower)


def triton_matmul_upper_tri(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    
    M, N = A.shape
    K = A.shape[1]
    
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = lambda meta: (tl.cdiv(M, meta["BLOCK_M"]) * tl.cdiv(N, meta["BLOCK_N"]),)
    
    matmul_upper_tri_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_upper_tri(A, B)