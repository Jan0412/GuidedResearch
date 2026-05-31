import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr, B_ptr, out_ptr,
    M, K,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_ptr = A_ptr + row_idx * K
    acc = tl.zeros((), dtype=tl.float32)
    
    for start in range(0, K, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < K
        a = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
        acc += tl.sum(a * b)
        
    tl.store(out_ptr + row_idx, acc)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    out = torch.empty((M, 1), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 1024
    grid = (M,)
    matvec_kernel[grid](A, B, out.view(-1), M, K, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)