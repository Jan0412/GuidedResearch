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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < (K - k)
        mask_am = offs_am < M
        mask_bn = offs_bn < N
        
        A = tl.load(A_ptrs, mask=(mask_am[:, None] & mask_k[None, :]), other=0.0)
        B = tl.load(B_ptrs, mask=(mask_k[:, None] & mask_bn[None, :]), other=0.0)
        
        accumulator += tl.dot(A, B)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    C_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_out = mask_am[:, None] & mask_bn[None, :]
    tl.store(C_ptrs, accumulator, mask=mask_out)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.shape[1] == B.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, N = B.shape
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    
    matmul_kernel[grid](
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
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)