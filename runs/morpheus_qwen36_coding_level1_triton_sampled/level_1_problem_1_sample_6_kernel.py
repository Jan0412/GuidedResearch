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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in tl.range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        
        acc += tl.dot(a, b)
        
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions for multiplication."
    
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K, N = B.shape
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_blocks_m, num_blocks_n, 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)