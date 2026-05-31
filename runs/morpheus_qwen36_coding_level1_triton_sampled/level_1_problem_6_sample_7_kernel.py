import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row_offsets < M
    col_mask = col_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_block = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)

        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn
        b_block = tl.load(b_ptrs, mask=col_mask[None, :], other=0.0)

        acc = tl.dot(a_block, b_block, acc)

    c_ptrs = C_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required."
    
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, "Inner dimensions must match."
    
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    num_warps = 4
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K1,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)