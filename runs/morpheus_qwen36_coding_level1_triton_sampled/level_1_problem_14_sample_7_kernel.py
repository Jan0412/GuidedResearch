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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    
    A = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)
    B = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=b_mask, other=0.0)
    
    C = tl.dot(A, B)
    
    tl.store(C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, C, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."
    
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        C = triton_matmul(A, B)
        return torch.triu(C)