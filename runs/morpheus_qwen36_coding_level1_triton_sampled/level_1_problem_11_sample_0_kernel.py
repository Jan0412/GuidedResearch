import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, L, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    mask_m = row_idx < M
    mask_k = col_idx < K
    mask = mask_m[:, None] & mask_k[None, :]
    
    C_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    for l in range(0, L, BLOCK_L):
        l_idx = l + tl.arange(0, BLOCK_L)
        mask_l = l_idx < L
        
        A_tile = tl.load(A_ptr + row_idx[:, None] * L + l_idx[None, :], 
                         mask=mask_m[:, None] & mask_l[None, :], other=0.0)
        B_tile = tl.load(B_ptr + l_idx[:, None] * K + col_idx[None, :], 
                         mask=mask_l[:, None] & mask_k[None, :], other=0.0)
        
        C_acc = tl.dot(A_tile, B_tile, allow_tf32=False)
        
    tl.store(C_ptr + row_idx[:, None] * K + col_idx[None, :], C_acc, mask=mask)


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous().to(torch.float32)
    B = B.contiguous().to(torch.float32)
    
    b, i, j, l = A.shape
    k = B.shape[1]
    M = b * i * j
    L = l
    K = k
    
    A_2d = A.reshape(M, L).contiguous()
    C_2d = torch.empty(M, K, dtype=torch.float32, device=A.device)
    
    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_L = 64
    
    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], (K + meta["BLOCK_K"] - 1) // meta["BLOCK_K"])
    
    matmul_kernel[grid](A_2d, B, C_2d, M, L, K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_L=BLOCK_L, num_warps=4, num_stages=3)
    
    return C_2d.reshape(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return triton_matmul(A, B)