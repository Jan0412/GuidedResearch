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
    
    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_ak = tl.arange(0, BLOCK_K)
    offsets_bk = tl.arange(0, BLOCK_K)
    
    mask_am = offsets_am < M
    mask_bn = offsets_bn < N
    mask_ak = offsets_ak < K
    mask_bk = offsets_bk < K
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offsets_am[:, None] * stride_am + offsets_ak[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(mask_am[:, None] & (offsets_ak[None, :] < K - k)), other=0.0)
        
        b_ptrs = B_ptr + offsets_bk[:, None] * stride_bk + (offsets_bn[None, :] + k) * stride_bn
        b = tl.load(b_ptrs, mask=(mask_bk[:, None] & (offsets_bn[None, :] < N - k)), other=0.0)
        
        acc += tl.dot(a, b)
        
    c_ptrs = C_ptr + offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, N = B.shape
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
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
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)