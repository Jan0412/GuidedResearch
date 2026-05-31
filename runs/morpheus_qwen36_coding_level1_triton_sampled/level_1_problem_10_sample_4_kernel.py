import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, out_ptr,
    N, M, K, L,
    stride_an, stride_am, stride_ak,
    stride_bk, stride_bl,
    stride_on, stride_om, stride_ol,
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    k_offsets = tl.arange(0, BLOCK_K)
    
    base_a = pid_n * stride_an + m_offsets[:, None] * stride_am
    base_b = l_offsets[None, :] * stride_bl
    
    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        
        mask_a = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        mask_b = (k_offsets[:, None] < K) & (l_offsets[None, :] < L)
        
        A_tile = tl.load(A_ptr + base_a + k_offsets[None, :] * stride_ak, mask=mask_a, other=0.0)
        B_tile = tl.load(B_ptr + base_b + k_offsets[:, None] * stride_bk, mask=mask_b, other=0.0)
        
        acc += tl.dot(A_tile, B_tile)
        
    out_offsets = pid_n * stride_on + m_offsets[:, None] * stride_om + l_offsets[None, :] * stride_ol
    mask_out = (m_offsets[:, None] < M) & (l_offsets[None, :] < L)
    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2, "Inner dimensions must match."
    
    out = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_L = 128
    BLOCK_K = 64
    
    grid = lambda meta: (
        N,
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (L + meta["BLOCK_L"] - 1) // meta["BLOCK_L"]
    )
    
    matmul_kernel[grid](
        A, B, out,
        N, M, K, L,
        M * K, K, 1,
        L, 1,
        M * L, L, 1,
        BLOCK_M=BLOCK_M, BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)