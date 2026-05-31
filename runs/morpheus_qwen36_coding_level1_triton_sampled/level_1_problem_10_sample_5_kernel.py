import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    N, M, K, L,
    stride_an, stride_am, stride_ak,
    stride_bk, stride_bl,
    stride_on, stride_om, stride_ol,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_K_TILES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    m_offsets = tl.arange(0, BLOCK_M)
    l_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    mask_m = m_offsets < M - pid_m * BLOCK_M
    mask_l = l_offsets < L - pid_l * BLOCK_N
    
    A_base = A_ptr + pid_n * stride_an + pid_m * stride_am
    B_base = B_ptr + pid_l * stride_bl
    Out_base = Out_ptr + pid_n * stride_on + pid_m * stride_om
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_tile in range(NUM_K_TILES):
        A_ptrs = A_base + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        mask_k = k_offsets < K - k_tile * BLOCK_K
        A_tile = tl.load(A_ptrs, mask=mask_k[None, :], other=0.0)
        
        B_ptrs = B_base + k_offsets[:, None] * stride_bk + l_offsets[None, :] * stride_bl
        mask_k = k_offsets < K - k_tile * BLOCK_K
        B_tile = tl.load(B_ptrs, mask=mask_k[:, None], other=0.0)
        
        acc += tl.dot(A_tile, B_tile)
        
    Out_ptrs = Out_base + m_offsets[:, None] * stride_om + l_offsets[None, :] * stride_ol
    mask_out = mask_m[:, None] & mask_l[None, :]
    tl.store(Out_ptrs, acc, mask=mask_out)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, "Inner dimensions must match."
    
    out = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    NUM_K_TILES = K // BLOCK_K
    
    stride_an = M * K
    stride_am = K
    stride_ak = 1
    
    stride_bk = L
    stride_bl = 1
    
    stride_on = M * L
    stride_om = L
    stride_ol = 1
    
    grid = (N, (M + BLOCK_M - 1) // BLOCK_M, (L + BLOCK_N - 1) // BLOCK_N)
    
    matmul_kernel[grid](
        A, B, out,
        N, M, K, L,
        stride_an, stride_am, stride_ak,
        stride_bk, stride_bl,
        stride_on, stride_om, stride_ol,
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_TILES
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_batched_matmul(A, B)