import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    A_ptr,
    out_ptr,
    M, N,
    s,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (M * N)
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    out = A * s
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scalar_mul(A: torch.Tensor, s: float):
    assert A.is_cuda
    A = A.contiguous()
    out = torch.empty_like(A)
    M, N = A.shape
    n_elements = M * N
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scalar_mul_kernel[grid](A, out, M, N, s, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scalar_mul(A, s)