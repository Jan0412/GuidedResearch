import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_T_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    
    mask_m = row_offsets < M
    mask_n = col_offsets < M
    
    # Load A block: shape (BLOCK_M, BLOCK_N)
    A_block = tl.load(
        A_ptr + row_offsets[:, None] * stride_am + tl.arange(0, BLOCK_N)[None, :] * stride_ak,
        mask=mask_m[:, None],
        other=0.0
    )
    
    # Load B_T block: shape (BLOCK_M, BLOCK_N)
    B_T_block = tl.load(
        B_T_ptr + col_offsets[:, None] * stride_bm + tl.arange(0, BLOCK_N)[None, :] * stride_bn,
        mask=mask_n[:, None],
        other=0.0
    )
    
    # Compute C block: C[i, j] = sum_k A[i, k] * B_T[j, k]
    # Equivalent to A_block @ B_T_block^T
    C_block = tl.dot(A_block, B_T_block, trans_b=True)
    
    # Store C block: shape (BLOCK_M, BLOCK_M)
    C_block_ptr = C_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
    tl.store(C_block_ptr, C_block, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, N = A.shape
    # B is N x M, so B.T is M x N
    B_T = B.T.contiguous()
    
    C = torch.empty(M, M, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 32  # N is fixed at 32
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (M + BLOCK_M - 1) // BLOCK_M)
    
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bm, stride_bn = B_T.stride(0), B_T.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    matmul_kernel[grid](
        A, B_T, C,
        M, N,
        stride_am, stride_ak,
        stride_bm, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)