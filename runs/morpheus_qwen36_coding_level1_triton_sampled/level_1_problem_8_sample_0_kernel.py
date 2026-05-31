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
    NUM_STAGES: tl.constexpr
):
    block_m_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)
    
    offs_am = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_k = offs_k < K
    
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_am[:, None] & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_bn[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions for matmul"
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    NUM_STAGES = 2
    
    num_cta_m = (M + BLOCK_M - 1) // BLOCK_M
    num_cta_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_cta_m, num_cta_n, 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)