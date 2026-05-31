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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    row_idx = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = row_idx < M
    mask_n = col_idx < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_idx < K
        
        a_offsets = row_idx[:, None] * stride_am + k_idx[None, :] * stride_ak
        mask_a = mask_m[:, None] & mask_k[None, :]
        A = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        
        b_offsets = k_idx[:, None] * stride_bn + col_idx[None, :] * stride_bk
        mask_b = mask_k[:, None] & mask_n[None, :]
        B = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)
        
        acc = tl.dot(A, B, allow_tf32=False)
        
    c_offsets = row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    tl.store(C_ptr + c_offsets, acc, mask=mask)


def triton_matmul(A: torch.Tensor, B_T: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B_T.is_cuda
    A = A.contiguous()
    B_T = B_T.contiguous()
    
    M, K = A.shape
    _, N = B_T.shape
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        1,
    )
    
    matmul_kernel[grid](
        A, B_T, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_T.stride(0), B_T.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        num_stages=2,
        num_warps=4,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B_T = B.T.contiguous()
        return triton_matmul(A, B_T)