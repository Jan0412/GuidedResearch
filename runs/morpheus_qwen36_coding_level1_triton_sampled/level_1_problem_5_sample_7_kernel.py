import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_kernel(
    A_ptr,
    out_ptr,
    M,
    N,
    s,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map program IDs to block indices
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    
    # Create row and column offsets for the current block
    row_offsets = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a 2D mask to handle boundary conditions
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    
    # Load input matrix A with masking
    A = tl.load(A_ptr + row_offsets[:, None] * N + col_offsets[None, :], mask=mask, other=0.0)
    
    # Perform scalar multiplication
    out = A * s
    
    # Store the result with masking
    tl.store(out_ptr + row_offsets[:, None] * N + col_offsets[None, :], out, mask=mask)


def triton_mul(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel for matrix-scalar multiplication.
    """
    assert A.is_cuda, "Input tensor must be on CUDA."
    A = A.contiguous()
    out = torch.empty_like(A)
    
    M, N = A.shape
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the kernel
    mul_kernel[(grid_m, grid_n)](
        A, out, M, N, s,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for matrix-scalar multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_mul(A, s)