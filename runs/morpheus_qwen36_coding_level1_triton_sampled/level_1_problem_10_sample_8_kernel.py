import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_vec_matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    N, M, K, L,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    batch_idx = tl.program_id(0)
    
    offset_a = batch_idx * K
    offset_out = batch_idx * L
    
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offset_a + k + tl.arange(0, BLOCK_K)
        mask_a = tl.arange(0, BLOCK_K) < (K - k)
        a_vals = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        offset_b_k = k * L
        b_ptrs = B_ptr + offset_b_k + tl.arange(0, BLOCK_K)[:, None] * L + tl.arange(0, BLOCK_L)[None, :]
        mask_b = (tl.arange(0, BLOCK_K)[:, None] < (K - k)) & (tl.arange(0, BLOCK_L)[None, :] < L)
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
        
    out_ptrs = Out_ptr + offset_out + tl.arange(0, BLOCK_L)
    mask_out = tl.arange(0, BLOCK_L) < L
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_batched_vec_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    _, L = B.shape
    
    Out = torch.empty(N, M, L, dtype=A.dtype, device=A.device)
    
    BLOCK_K = 64
    BLOCK_L = 64
    
    grid = (N * M,)
    
    batched_vec_matmul_kernel[grid](
        A, B, Out,
        N, M, K, L,
        BLOCK_K=BLOCK_K,
        BLOCK_L=BLOCK_L
    )
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A, B):
        return triton_batched_vec_matmul(A, B)