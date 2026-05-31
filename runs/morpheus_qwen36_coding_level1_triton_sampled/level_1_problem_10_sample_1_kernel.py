import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    N, M, K, L,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_l = offs_l < L
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + pid * M * K + offs_m[:, None] * K + offs_k[None, :] + k
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + offs_k[:, None] * L + offs_l[None, :] + k * L
        b_mask = mask_k[:, None] & mask_l[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b)

    c_ptrs = Out_ptr + pid * M * L + offs_m[:, None] * L + offs_l[None, :]
    c_mask = mask_m[:, None] & mask_l[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    _, L = B.shape
    assert K == B.shape[0]
    
    Out = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_K = 128
    BLOCK_L = 128
    
    grid = (N, (M + BLOCK_M - 1) // BLOCK_M, (L + BLOCK_L - 1) // BLOCK_L)
    
    matmul_kernel[grid](
        A, B, Out,
        N, M, K, L,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_L=BLOCK_L
    )
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return triton_matmul(A, B)