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
    pid = tl.program_id(0)
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = BLOCK_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * BLOCK_SIZE_M
    group_size_m = min(num_pid_m - group_id, BLOCK_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % (group_size_m * num_pid_n)) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                    mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn, 
                    mask=offs_bn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = acc.to(tl.float32)
    tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, 
             c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
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