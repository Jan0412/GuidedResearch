import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_T_ptr, C_ptr,
    M, N,
    stride_a_m, stride_a_n,
    stride_b_m, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    
    mask_m = off_m < M
    mask_n = off_n < M
    mask_k = off_k < N
    
    A_ptrs = A_ptr + off_m[:, None] * stride_a_m + off_k[None, :] * stride_a_n
    B_T_ptrs = B_T_ptr + off_n[:, None] * stride_b_m + off_k[None, :] * stride_b_n
    
    A = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    B_T = tl.load(B_T_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
    
    C_block = tl.dot(A, B_T.T)
    
    C_ptrs = C_ptr + off_m[:, None] * stride_c_m + off_n[None, :] * stride_c_n
    tl.store(C_ptrs, C_block, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B_T: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B_T.is_cuda
    A = A.contiguous()
    B_T = B_T.contiguous()
    
    M, K = A.shape
    _, N = B_T.shape
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A, B_T, C,
        M, K,
        A.stride(0), A.stride(1),
        B_T.stride(0), B_T.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        B_T = B.T.contiguous()
        return triton_matmul(A, B_T)