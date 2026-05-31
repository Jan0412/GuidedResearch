import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def row_scale_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    
    # Load the diagonal element for this row once
    A_val = tl.load(A_ptr + pid)
    
    # Number of column blocks needed to cover M
    num_col_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Column offsets for the current block
    col_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    for block in range(num_col_blocks):
        col_start = block * BLOCK_SIZE_M
        col_idx = col_start + col_offsets
        mask = col_idx < M
        
        # Load B[row, col]
        B_val = tl.load(B_ptr + pid * M + col_idx, mask=mask, other=0.0)
        
        # Scale by A[row] and store result
        C_val = A_val * B_val
        tl.store(C_ptr + pid * M + col_idx, C_val, mask=mask)


def triton_row_scale(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    C = torch.empty_like(B)
    
    BLOCK_SIZE_M = 128
    
    # Launch one thread block per row
    grid = (N,)
    row_scale_kernel[grid](A, B, C, N, M, BLOCK_SIZE_M)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_row_scale(A, B)