import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= N:
        return
        
    a_val = tl.load(A_ptr + pid)
    row_offset = pid * M
    
    for start_col in range(0, M, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        b_vals = tl.load(B_ptr + row_offset + offsets, mask=mask, other=0.0)
        c_vals = a_val * b_vals
        tl.store(C_ptr + row_offset + offsets, c_vals, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    C = torch.empty_like(B)
    
    BLOCK_SIZE = 128
    grid = (N, 1)
    
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)


M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []