import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, out_ptr,
    N, M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid coordinates
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    
    # Row and column offsets
    rows = row_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = col_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_rows = rows < N
    mask_cols = cols < M
    mask = mask_rows[:, None] & mask_cols[None, :]
    
    # Load A and B
    # A is shape (N,), B is shape (N, M)
    a = tl.load(A_ptr + rows, mask=mask_rows, other=0.0)
    b = tl.load(B_ptr + rows[:, None] * M + cols[None, :], mask=mask, other=0.0)
    
    # Multiply: each row i of B is scaled by A[i]
    out = a[:, None] * b
    
    # Store result
    tl.store(out_ptr + rows[:, None] * M + cols[None, :], out, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = diag(A) @ B using a custom Triton kernel.
    Equivalent to scaling each row of B by the corresponding element in A.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D."
    assert B.dim() == 2, "B must be 2D."
    assert A.shape[0] == B.shape[0], "A and B must have matching first dimension."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = A.shape[0], B.shape[1]
    out = torch.empty((N, M), dtype=B.dtype, device=B.device)
    
    # Tunable block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Grid calculation
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_N))
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)