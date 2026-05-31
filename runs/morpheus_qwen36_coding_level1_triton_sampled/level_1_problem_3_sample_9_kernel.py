import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    m, k, n, batch_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < batch_size:
        a_offset = pid * m * k
        b_offset = pid * k * n
        c_offset = pid * m * n
        
        a_ptrs = A_ptr + a_offset
        b_ptrs = B_ptr + b_offset
        c_ptrs = C_ptr + c_offset
        
        # tl.dot handles tiling and tensor core usage internally
        c = tl.dot(a_ptrs, b_ptrs, m, k, n)
        tl.store(c_ptrs, c)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    # Tunable block sizes for optimal occupancy and cache usage
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = lambda meta: (batch_size,)
    bmm_kernel[grid](
        A, B, C,
        m, k, n, batch_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)