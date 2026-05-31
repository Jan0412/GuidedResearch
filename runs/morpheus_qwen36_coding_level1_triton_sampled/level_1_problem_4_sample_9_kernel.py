import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Matrix-Vector multiplication: C = A * B
    A: (M, K)
    B: (K, 1)
    C: (M, 1)
    """
    row_idx = tl.program_id(0)
    # Pointer to the start of the row in A
    row_start_ptr = A_ptr + row_idx * K
    # Pointer to the output element
    out_ptr = C_ptr + row_idx
    
    acc = 0.0
    # Loop over blocks of K
    for block_start in range(0, K, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < K
        
        # Load block from A row
        a_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Load block from B vector
        b_block = tl.load(B_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(a_block * b_block)
        
    tl.store(out_ptr, acc)


def triton_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton GEMV kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    out = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # BLOCK_SIZE should be a power of 2 and a divisor of K for optimal performance.
    # K = 1048576 = 2^20, so 4096 is a good choice.
    BLOCK_SIZE = 4096
    
    grid = (M,)
    
    gemv_kernel[grid](A, B, out, M, K, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_gemv(A, B)