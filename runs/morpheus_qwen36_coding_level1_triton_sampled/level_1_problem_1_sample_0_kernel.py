import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_k = offs_k < K
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_am[:, None] * N + (offs_k + k)[None, :], mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_ptr + (offs_k + k)[:, None] * N + offs_bn[None, :], mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        acc += tl.dot(a, b)
        
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(C_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions"
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)

def get_inputs():
    N = 2048 * 2
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []