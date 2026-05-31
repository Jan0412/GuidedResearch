import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a_n, stride_a_m, stride_a_k,
    stride_b_k, stride_b_l,
    stride_c_n, stride_c_m, stride_c_l,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offsets_k = tl.arange(0, BLOCK_K)

    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    
    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    
    for k in range(num_k_blocks):
        offsets_a = pid_n * stride_a_n + offsets_m[:, None] * stride_a_m + offsets_k[None, :] * stride_a_k
        mask_a = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        a = tl.load(A_ptr + offsets_a, mask=mask_a, other=0.0)
        
        offsets_b = offsets_k[:, None] * stride_b_k + offsets_l[None, :] * stride_b_l
        mask_b = (offsets_k[:, None] < K) & (offsets_l[None, :] < L)
        b = tl.load(B_ptr + offsets_b, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
    offsets_c = pid_n * stride_c_n + offsets_m[:, None] * stride_c_m + offsets_l[None, :] * stride_c_l
    mask_c = (offsets_m[:, None] < M) & (offsets_l[None, :] < L)
    tl.store(C_ptr + offsets_c, acc, mask=mask_c)

def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2
    
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_L = 64
    
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_l_blocks = (L + BLOCK_L - 1) // BLOCK_L
    
    grid = (N, num_m_blocks, num_l_blocks)
    
    stride_a_n = M * K
    stride_a_m = K
    stride_a_k = 1
    
    stride_b_k = L
    stride_b_l = 1
    
    stride_c_n = M * L
    stride_c_m = L
    stride_c_l = 1
    
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_a_n, stride_a_m, stride_a_k,
        stride_b_k, stride_b_l,
        stride_c_n, stride_c_m, stride_c_l,
        BLOCK_M, BLOCK_K, BLOCK_L
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return triton_matmul(A, B)