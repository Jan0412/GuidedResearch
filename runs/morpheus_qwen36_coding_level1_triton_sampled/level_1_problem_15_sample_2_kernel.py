import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_tril_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Block indices
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)

    # Check if block is in upper triangle (fully invalid)
    # Block covers rows [block_idx_m * BLOCK_M, (block_idx_m + 1) * BLOCK_M)
    # and cols [block_idx_n * BLOCK_N, (block_idx_n + 1) * BLOCK_N)
    # We need at least one element where row >= col.
    # This requires block_idx_m * BLOCK_M < (block_idx_n + 1) * BLOCK_N
    if block_idx_m * BLOCK_M >= (block_idx_n + 1) * BLOCK_N:
        return

    # Offsets
    offs_m = block_idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create masks for loading A and B, and for storing C
    # A is valid where k <= i
    mask_A = offs_k[None, :] <= offs_m[:, None]
    # B is valid where k <= j
    mask_B = offs_k[:, None] <= offs_n[None, :]
    # C is valid where i >= j
    mask_C = offs_m[:, None] >= offs_n[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, M, BLOCK_K):
        # Load tiles with masking
        A_tile = tl.load(A_ptr + offs_m[:, None] * M + (k + offs_k[None, :]), 
                         mask=mask_A, other=0.0)
        B_tile = tl.load(B_ptr + (k + offs_k[:, None]) * N + offs_n[None, :], 
                         mask=mask_B, other=0.0)
        
        # Matrix multiply
        acc += tl.dot(A_tile, B_tile)

    # Store result with masking
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_C)


def triton_matmul_tril(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = tril(A @ B) using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape
    M = A.shape[0]
    
    # Output tensor initialized to zeros
    C = torch.zeros((M, M), dtype=torch.float32, device='cuda')
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))
    
    # Launch kernel
    matmul_tril_kernel[grid](
        A, B, C, M, M,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_tril(A, B)