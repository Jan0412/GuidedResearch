import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_offsets = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    acc = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        col_offsets = k + tl.arange(0, BLOCK_K)
        col_mask = col_offsets < K

        A_block = tl.load(A_ptr + row_offsets[:, None] * K + col_offsets[None, :],
                          mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        B_block = tl.load(B_ptr + col_offsets, mask=col_mask, other=0.0).reshape(BLOCK_K, 1)

        acc = acc + tl.dot(A_block, B_block, out_dtype=tl.float32)

    C_block = acc[:, 0]
    tl.store(C_ptr + row_offsets, C_block, mask=row_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    assert B.shape == (K, 1)

    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)

    BLOCK_M = 128
    BLOCK_K = 256

    grid = ((M + BLOCK_M - 1) // BLOCK_M, 1, 1)
    matmul_kernel[grid](A, B, C, M, K, BLOCK_M, BLOCK_K)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)