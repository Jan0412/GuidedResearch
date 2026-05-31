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

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k_tiles):
        k_start = k * BLOCK_K

        # Load tile from A
        a_rows = rows
        a_cols = tl.arange(0, BLOCK_K) + k_start
        mask_a = (a_rows[:, None] < M) & (a_cols[None, :] < K)
        a_tile = tl.load(A_ptr + a_rows[:, None] * stride_am + a_cols[None, :] * stride_ak,
                         mask=mask_a, other=0.0)

        # Load tile from B
        b_rows = tl.arange(0, BLOCK_K) + k_start
        b_cols = cols
        mask_b = (b_rows[:, None] < K) & (b_cols[None, :] < N)
        b_tile = tl.load(B_ptr + b_rows[:, None] * stride_bk + b_cols[None, :] * stride_bn,
                         mask=mask_b, other=0.0)

        # Perform tile-level matrix multiplication
        acc += tl.dot(a_tile, b_tile)

    # Store result with masking for irregular boundaries
    mask_c = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K dimensions must match: {K} vs {K2}"
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    grid = (num_m_tiles, num_n_tiles, 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)