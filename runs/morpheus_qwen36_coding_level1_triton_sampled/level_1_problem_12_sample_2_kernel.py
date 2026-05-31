import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def row_scale_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Load the scaling factor for this row
    a_val = tl.load(A_ptr + row_idx)
    
    # Compute offsets for B and C
    row_offset = row_idx * M
    col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < M
    
    b_ptr = B_ptr + row_offset + col_offsets
    c_ptr = C_ptr + row_offset + col_offsets
    
    # Load B block and multiply
    b_val = tl.load(b_ptr, mask=mask, other=0.0)
    c_val = a_val * b_val
    
    # Store result
    tl.store(c_ptr, c_val, mask=mask)

def triton_row_scale(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N, M = A.shape[0], B.shape[1]
    C = torch.empty_like(B)
    
    BLOCK_SIZE = 128
    grid = (N, (M + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    row_scale_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_row_scale(A, B)

M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []