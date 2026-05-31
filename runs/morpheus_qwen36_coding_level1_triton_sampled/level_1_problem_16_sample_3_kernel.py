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
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A_tile = tl.load(A_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am,
                         mask=(mask_k[:, None] & mask_m[None, :]), other=0.0)
        B_tile = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                         mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        acc += tl.dot(A_tile, B_tile, transpose_a=True)

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(mask_m[:, None] & mask_n[None, :]))

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    num_m = (M + 127) // 128
    num_n = (N + 127) // 128
    grid = (num_m * num_n, 1, 1)
    
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)