import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(
    A_ptr, s, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    num_elements = M * N
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    A_vals = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    out_vals = A_vals * s
    tl.store(out_ptr + offsets, out_vals, mask=mask)

def triton_scalar_mul(A: torch.Tensor, s: float) -> torch.Tensor:
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    out = torch.empty_like(A)
    num_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scalar_mul_kernel[grid](A, s, out, A.shape[0], A.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scalar_mul(A, s)