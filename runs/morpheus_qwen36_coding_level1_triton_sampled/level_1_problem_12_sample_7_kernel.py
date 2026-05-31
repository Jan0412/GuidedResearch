import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx < N:
        a_val = tl.load(A_ptr + row_idx)
        col_off = 0
        while col_off < M:
            offsets = col_off + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            b_vals = tl.load(B_ptr + row_idx * M + offsets, mask=mask, other=0.0)
            c_vals = a_val * b_vals
            tl.store(C_ptr + row_idx * M + offsets, c_vals, mask=mask)
            col_off += BLOCK_SIZE


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    M = B.shape[1]
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)
    BLOCK_SIZE = 128
    grid = (N,)
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_diag_matmul(A, B)