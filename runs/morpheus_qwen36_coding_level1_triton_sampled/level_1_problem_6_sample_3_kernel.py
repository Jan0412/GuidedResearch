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
    
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_blocks):
        offsets_k = tl.arange(0, BLOCK_K)
        a_block = tl.load(A_ptr + offsets_k)
        b_block = tl.load(B_ptr + (k * BLOCK_K) * stride_bk + offsets_k)
        acc = tl.dot(a_block, b_block, acc)
        
    tl.store(C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64
    
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)