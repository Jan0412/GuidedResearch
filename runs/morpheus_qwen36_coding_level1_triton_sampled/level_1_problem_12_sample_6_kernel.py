import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, out_ptr, N, M,
    BLOCK_SIZE_COL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Load the diagonal element for this row (constant across the row)
    a_val = tl.load(A_ptr + row_idx)
    
    # Calculate column offsets for the current block
    col_offsets = col_block_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    mask = col_offsets < M
    
    # Load the corresponding row segment from B
    b_segment = tl.load(B_ptr + row_idx * M + col_offsets, mask=mask, other=0.0)
    
    # Compute element-wise multiplication: C[i, j] = A[i] * B[i, j]
    out_segment = a_val * b_segment
    
    # Store the result
    tl.store(out_ptr + row_idx * M + col_offsets, out_segment, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = A.shape[0], B.shape[1]
    out = torch.empty((N, M), dtype=B.dtype, device=B.device)
    
    BLOCK_SIZE_COL = 128
    num_col_blocks = (M + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    
    grid = (N, num_col_blocks)
    
    diag_matmul_kernel[grid](A, B, out, N, M, BLOCK_SIZE_COL=BLOCK_SIZE_COL)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)