import torch
import torch.nn as nn
import triton
import triton.language as tl

M = 16384 * 2
N = 16 * 2

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program ID gives the block coordinates
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    
    # Generate row and column offsets for the current block
    rows = tl.arange(0, BLOCK_M)[:, None] + row_block_id * BLOCK_M
    cols = tl.arange(0, BLOCK_N)[None, :] + col_block_id * BLOCK_N
    
    # Generate k offsets (inner dimension)
    k_offsets = tl.arange(0, N)[:, None]
    
    # Compute global offsets for A, B, and C
    # A is (M, N), B is (N, M), C is (M, M)
    A_offsets = rows * N + k_offsets
    B_offsets = k_offsets * M + cols
    C_offsets = rows * M + cols
    
    # Load tiles from global memory
    # A_tile shape: (BLOCK_M, N)
    # B_tile shape: (N, BLOCK_N)
    A_tile = tl.load(A_ptr + A_offsets)
    B_tile = tl.load(B_ptr + B_offsets)
    
    # Perform matrix multiplication for the block
    # C_tile shape: (BLOCK_M, BLOCK_N)
    C_tile = tl.dot(A_tile, B_tile)
    
    # Store result to global memory
    tl.store(C_ptr + C_offsets, C_tile)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for matrix multiplication C = A @ B.
    Optimized for cases where A is (M, N) and B is (N, M) with small N.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul."
    
    M_in, K = A.shape
    K_out, N_out = B.shape
    
    assert K == K_out, "Inner dimensions must match."
    
    C = torch.empty((M_in, N_out), dtype=A.dtype, device=A.device)
    
    # Configuration parameters
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Ensure dimensions are divisible by block sizes for simplicity
    # In production, padding might be needed, but here M and N are chosen to be multiples.
    assert M_in % BLOCK_M == 0, f"M ({M_in}) must be divisible by BLOCK_M ({BLOCK_M})"
    assert N_out % BLOCK_N == 0, f"N ({N_out}) must be divisible by BLOCK_N ({BLOCK_N})"
    
    # Grid configuration
    grid = (M_in // BLOCK_M, N_out // BLOCK_N)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M_in, N_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)


def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]


def get_init_inputs():
    return []