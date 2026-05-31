import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    offsets_m = offset_m + tl.arange(0, BLOCK_M)
    offsets_n = offset_n + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    
    # Load A tile: shape (BLOCK_M, BLOCK_K)
    # A is (M, K). Row stride is K.
    A_tile = tl.load(A_ptr + offsets_m[:, None] * K + offsets_k[None, :], 
                     mask=mask_m[:, None] & mask_k[None, :], 
                     other=0.0)
    
    # Load B tile: shape (BLOCK_N, BLOCK_K)
    # B is (N, K) in kernel context (transposed input)
    B_tile = tl.load(B_ptr + offsets_n[:, None] * K + offsets_k[None, :], 
                     mask=mask_n[:, None] & mask_k[None, :], 
                     other=0.0)
    
    # Compute dot product
    C_tile = tl.dot(A_tile, B_tile.T)
    
    # Store C tile
    tl.store(C_ptr + offsets_m[:, None] * N + offsets_n[None, :], 
             C_tile, 
             mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    
    # B is (K, M) in original input.
    # We transpose B to (M, K) for efficient memory access in the kernel.
    B_T = B.T.contiguous()
    
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N
    )
    
    matmul_kernel[grid](A, B_T, C, M, N, K, 
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)