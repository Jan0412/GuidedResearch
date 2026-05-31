import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A = tl.load(A_ptr + offs_am[:, None] * N + offs_k[None, :], 
                mask=(offs_am[:, None] < N) & (offs_k[None, :] < N), other=0.0)
    B = tl.load(B_ptr + offs_k[:, None] * N + offs_bn[None, :], 
                mask=(offs_k[:, None] < N) & (offs_bn[None, :] < N), other=0.0)

    C = tl.dot(A, B)

    offs_cm = offs_am[:, None]
    offs_cn = offs_bn[None, :]
    tl.store(C_ptr + offs_cm * N + offs_cn, C, mask=(offs_cm < N) & (offs_cn < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.empty_like(A)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    num_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_m * num_n,)
    
    matmul_kernel[grid](A, B, C, N, 
                        BLOCK_SIZE_M=BLOCK_SIZE_M, 
                        BLOCK_SIZE_N=BLOCK_SIZE_N, 
                        BLOCK_SIZE_K=BLOCK_SIZE_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)