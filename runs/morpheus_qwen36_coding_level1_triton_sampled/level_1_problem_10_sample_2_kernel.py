import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_am, stride_ak, stride_an,
    stride_bk, stride_bl,
    stride_cm, stride_cl, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    n_idx = tl.program_id(2)
    
    # Base pointers for the current tile
    a_offset = batch_idx * stride_am + m_idx * BLOCK_M * stride_ak
    b_offset = n_idx * BLOCK_N * stride_bl
    
    # Create offsets for loading tiles
    a_row_offsets = tl.arange(0, BLOCK_M)
    a_col_offsets = tl.arange(0, BLOCK_K)
    a_offsets = a_offset + a_row_offsets[:, None] * stride_ak + a_col_offsets[None, :] * stride_an
    
    b_row_offsets = tl.arange(0, BLOCK_K)
    b_col_offsets = tl.arange(0, BLOCK_N)
    b_offsets = b_offset + b_row_offsets[:, None] * stride_bk + b_col_offsets[None, :] * stride_bl
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Bounds checking masks for M and L
        row_mask = (m_idx * BLOCK_M + a_row_offsets) < M
        col_mask = (n_idx * BLOCK_N + b_col_offsets) < L
        
        # Load tiles (K is fully covered by BLOCK_K loops, so no mask needed for K)
        a = tl.load(a_offsets, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_offsets, mask=col_mask[None, :], other=0.0)
        
        # Matrix multiplication tile
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_offsets += BLOCK_K * stride_an
        b_offsets += BLOCK_K * stride_bk
        
    # Store result
    c_offset = batch_idx * stride_cn + m_idx * BLOCK_M * stride_cm + n_idx * BLOCK_N * stride_cl
    c_row_offsets = tl.arange(0, BLOCK_M)
    c_col_offsets = tl.arange(0, BLOCK_N)
    c_offsets = c_offset + c_row_offsets[:, None] * stride_cm + c_col_offsets[None, :] * stride_cl
    
    c_mask = (m_idx * BLOCK_M + c_row_offsets) < M
    c_mask &= (n_idx * BLOCK_N + c_col_offsets) < L
    
    tl.store(c_offsets, acc, mask=c_mask)


def triton_matmul(A, B):
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b
    
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Tunable block sizes optimized for the given dimensions
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Grid configuration: (batch, M_blocks, L_blocks)
    grid = (N, M // BLOCK_M, L // BLOCK_N)
    
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)