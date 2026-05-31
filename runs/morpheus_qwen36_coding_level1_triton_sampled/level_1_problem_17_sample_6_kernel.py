import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid coordinates for the block
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)

    # Base offsets for the current block
    m_off = m_block * BLOCK_M
    n_off = n_block * BLOCK_N

    # Create row and column indices for the block
    m_indices = m_off + tl.arange(0, BLOCK_M)
    n_indices = n_off + tl.arange(0, BLOCK_N)
    k_indices = tl.arange(0, BLOCK_K)

    # Mask for valid elements
    mask_m = m_indices < M
    mask_n = n_indices < N
    mask_k = k_indices < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tile of A
        a_offsets = m_indices[:, None] * stride_am + (k + k_indices[None, :]) * stride_ak
        mask_a = mask_m[:, None] & (k + k_indices[None, :] < K)
        A_tile = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)

        # Load tile of B
        b_offsets = (n_indices[:, None] + k) * stride_bn + k_indices[None, :] * stride_bk
        mask_b = mask_n[:, None] & (k + k_indices[None, :] < K)
        B_tile = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)

        # Transpose B tile to align with A for dot product
        B_tile_T = tl.trans(B_tile)

        # Compute partial dot product
        acc += tl.dot(A_tile, B_tile_T)

    # Store result
    c_offsets = m_indices[:, None] * stride_cm + n_indices[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "Incompatible dimensions for matrix multiplication."

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)