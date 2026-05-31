import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    # Load the diagonal element for this row
    A_val = tl.load(A_ptr + row_idx)
    
    # Process columns in blocks
    for start_col in range(0, M, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < M
        
        # Load block from B
        B_block = tl.load(B_ptr + row_idx * M + col_offsets, mask=mask, other=0.0)
        
        # Scale by A_val
        C_block = A_val * B_block
        
        # Store block to C
        tl.store(C_ptr + row_idx * M + col_offsets, C_block, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = A.shape[0], B.shape[1]
    C = torch.empty((N, M), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 1024
    
    grid = (N,)
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)