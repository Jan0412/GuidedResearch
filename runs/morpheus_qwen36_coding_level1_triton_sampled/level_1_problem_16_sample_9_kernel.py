import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    stride_A_k, stride_A_i,
    stride_B_k, stride_B_j,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create index grids for the tile
    k_idx = tl.arange(0, BLOCK_K)
    i_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    j_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds checking
    mask_k = k_idx[:, None] < K
    mask_i = i_idx[None, :] < M
    mask_j = j_idx[None, :] < N

    # Offsets for A and B
    # A has shape (K, M), strides (stride_A_k, stride_A_i)
    # B has shape (K, N), strides (stride_B_k, stride_B_j)
    A_offsets = k_idx[:, None] * stride_A_k + i_idx[None, :] * stride_A_i
    B_offsets = k_idx[:, None] * stride_B_k + j_idx[None, :] * stride_B_j

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles
        A_tile = tl.load(A_ptr + A_offsets, mask=mask_k & mask_i, other=0.0)
        B_tile = tl.load(B_ptr + B_offsets, mask=mask_k & mask_j, other=0.0)

        # Compute partial dot product
        # A_tile is (BLOCK_K, BLOCK_M), B_tile is (BLOCK_K, BLOCK_N)
        # We need sum_k A[k, i] * B[k, j], which is A_tile.T @ B_tile
        acc += tl.dot(A_tile.T, B_tile)

        # Update offsets for next block of K
        A_offsets += BLOCK_K * stride_A_k
        B_offsets += BLOCK_K * stride_B_k

    # Store result
    C_offsets = i_idx[:, None] * stride_C_m + j_idx[None, :] * stride_C_n
    tl.store(C_ptr + C_offsets, acc, mask=mask_i & mask_j)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M = A.shape[1]  # A is (K, M), A.T is (M, K)
    K = A.shape[0]
    N = B.shape[1]  # B is (K, N)

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Strides
    stride_A_k, stride_A_i = A.stride()
    stride_B_k, stride_B_j = B.stride()
    stride_C_m, stride_C_n = C.stride()

    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K, N,
        stride_A_k, stride_A_i,
        stride_B_k, stride_B_j,
        stride_C_m, stride_C_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)