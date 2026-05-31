import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    rows = block_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = block_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_rows = rows < M
    mask_cols = cols < M
    mask_out = mask_rows[:, None] & mask_cols[None, :] & (rows >= cols)
    mask_upper = ~mask_out
    
    k_start = block_j * BLOCK_SIZE
    k_end = (block_i + 1) * BLOCK_SIZE
    k_len = k_end - k_start
    
    k = k_start + tl.arange(0, k_len)
    mask_k = k < M
    
    if block_i < block_j:
        c_ptrs = C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
        tl.store(c_ptrs, 0.0, mask=mask_upper)
        return

    a_ptrs = A_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak
    a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
    
    b_ptrs = B_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn
    b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
    
    c = tl.dot(a, b)
    
    c_ptrs = C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_out)


def triton_tril_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, N = A.shape
    assert A.shape == B.shape
    C = torch.zeros_like(A)
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    tril_matmul_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_tril_matmul(A, B)