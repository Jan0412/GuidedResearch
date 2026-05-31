import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Tile boundary masks
    mask_m = rows < N
    mask_n = cols < N

    # Accumulator for FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled reduction over K dimension
    for k in range(0, N, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load A tile: valid only where k <= i (lower triangular property)
        a_mask = mask_m[:, None] & (k_offsets[None, :] <= rows[:, None])
        a_ptrs = A_ptr + rows[:, None] * N + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: valid only where k >= j (lower triangular property)
        b_mask = mask_n[None, :] & (k_offsets[:, None] >= cols[None, :])
        b_ptrs = B_ptr + k_offsets[:, None] * N + cols[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply tile
        acc += tl.dot(a_tile, b_tile)

    # Store result only for lower triangle elements
    out_mask = mask_m[:, None] & mask_n[None, :] & (rows[:, None] >= cols[None, :])
    tl.store(C_ptr + rows[:, None] * N + cols[None, :], acc, mask=out_mask)


def triton_lower_tri_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the custom Triton kernel for lower triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    N = A.shape[0]
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = ((N + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    lower_tri_matmul_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # Exploit lower triangular structure to skip upper triangle computation
        # and remove redundant torch.tril operation
        return triton_lower_tri_matmul(A, B)