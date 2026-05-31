import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for lower triangular part: only compute/store where row >= col
    mask = row_offsets[:, None] >= col_offsets[None, :]

    # Accumulator for FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k = (N + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Compute pointers for A and B tiles
        a_ptr = A_ptr + row_offsets[:, None] * N + k_offsets[None, :]
        b_ptr = B_ptr + k_offsets[:, None] * N + col_offsets[None, :]

        # Masks for boundary conditions
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < N)
        b_mask = (k_offsets[:, None] < N) & (col_offsets[None, :] < N)

        a = tl.load(a_ptr, mask=a_mask, other=0.0)
        b = tl.load(b_ptr, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store result only for lower triangular elements
    c_ptr = C_ptr + row_offsets[:, None] * N + col_offsets[None, :]
    tl.store(c_ptr, acc, mask=mask)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape
    M, N = A.shape

    # Initialize with zeros to match torch.tril behavior for upper triangle
    C = torch.zeros_like(A)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_rows = (M + BLOCK_M - 1) // BLOCK_M
    num_cols = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_rows, num_cols)

    tril_matmul_kernel[grid](
        A, B, C,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_tril_matmul(A, B)