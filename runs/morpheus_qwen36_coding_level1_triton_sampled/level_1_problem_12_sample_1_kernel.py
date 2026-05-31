import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_mul_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one row of the output matrix
    # Load the diagonal scaling factor for this row
    a_val = tl.load(A_ptr + pid)
    
    # Generate column offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    
    # Load the row from B
    b_vals = tl.load(B_ptr + pid * M + offsets, mask=mask, other=0.0)
    
    # Multiply row by diagonal element
    out_vals = a_val * b_vals
    
    # Store result
    tl.store(out_ptr + pid * M + offsets, out_vals, mask=mask)

def triton_diag_mul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty_like(B)
    N, M = A.shape[0], B.shape[1]
    BLOCK_SIZE = 128
    grid = (N,)
    diag_mul_kernel[grid](A, B, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return triton_diag_mul(A, B)