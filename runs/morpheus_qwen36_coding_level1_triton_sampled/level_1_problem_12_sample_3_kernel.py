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
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Map linear offset to 2D coordinates (i, j)
    i = offsets // M
    j = offsets % M
    
    # Load diagonal element and corresponding row element
    A_val = tl.load(A_ptr + i, mask=mask, other=0.0)
    B_val = tl.load(B_ptr + i * M + j, mask=mask, other=0.0)
    
    # Compute element-wise product
    out = A_val * B_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_diag_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    n_elements = N * M
    
    out = torch.empty((N, M), dtype=A.dtype, device=A.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    diag_mul_kernel[grid](A, B, out, N, M, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A, B):
        return triton_diag_mul(A, B)