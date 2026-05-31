import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Create masks for loading A and B blocks
    mask_a = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
    mask_b = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

    # Load blocks from A and B
    A_block = tl.load(A_ptr + row_offsets[:, None] * M + k_offsets[None, :], mask=mask_a, other=0.0)
    B_block = tl.load(B_ptr + k_offsets[:, None] * M + col_offsets[None, :], mask=mask_b, other=0.0)

    # Perform block matrix multiplication
    C_block = tl.dot(A_block, B_block, allow_tf32=False)

    # Mask to keep only lower triangular elements
    mask_out = row_offsets[:, None] >= col_offsets[None, :]

    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * M + col_offsets[None, :], C_block, mask=mask_out)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, N = A.shape[0], B.shape[1]
    K = A.shape[1]
    C = torch.empty(M, N, dtype=torch.float32, device='cuda')

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    tril_matmul_kernel[grid](A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_tril_matmul(A, B)