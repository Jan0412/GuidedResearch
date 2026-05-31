import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit(num_warps=4, num_stages=2)
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A_block = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
                          mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        A_block = tl.trans(A_block)
        
        B_block = tl.load(B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                          mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        B_block = tl.trans(B_block)
        
        acc = tl.dot(A_block, B_block)

    offs_cm = pid_m * BLOCK_M + offs_m
    offs_cn = pid_n * BLOCK_N + offs_n
    
    tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    N, _ = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device='cuda')
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 16
    num_warps = 4
    
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    
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
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)