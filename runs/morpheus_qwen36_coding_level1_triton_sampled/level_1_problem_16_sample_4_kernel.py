import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    
    offs_am = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_block in range(0, K, BLOCK_K):
        k_offsets = k_block + offs_k
        
        # Load A block: A is (K, M), we want A.T block (M, K)
        # A.T[m, k] = A[k, m]
        a_ptrs = A_ptr + k_offsets[None, :] * stride_ak + offs_am[:, None] * stride_am
        a_block = tl.load(a_ptrs, mask=k_offsets[None, :] < K, other=0.0)
        
        # Load B block: B is (K, N)
        b_ptrs = B_ptr + k_offsets[None, :] * stride_bk + offs_bn[:, None] * stride_bn
        b_block = tl.load(b_ptrs, mask=k_offsets[None, :] < K, other=0.0)
        
        acc += tl.dot(a_block, b_block)
        
    # Store C block: C is (M, N)
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_block = acc
    tl.store(c_ptrs, c_block, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match"
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        1,
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)