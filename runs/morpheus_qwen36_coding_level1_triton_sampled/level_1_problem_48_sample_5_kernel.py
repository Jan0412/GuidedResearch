import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr, out_ptr,
    B, N, M,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output element corresponding to (b, m)
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    # Base pointer for the row we are reducing
    row_start = b * N * M + m
    
    acc = 0.0
    # Loop over the reduction dimension in chunks
    for start in range(0, N, BLOCK_SIZE):
        offsets = row_start + (start + tl.arange(0, BLOCK_SIZE)) * M
        mask = (start + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x)

    # Compute mean
    out_val = acc / N
    tl.store(out_ptr + pid, out_val)


def triton_mean_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    B, N, M = x.shape
    out = torch.empty((B, M), dtype=x.dtype, device=x.device)
    
    # Tunable block size for the reduction dimension
    BLOCK_SIZE = 1024
    grid = (B * M,)
    
    mean_reduce_kernel[grid](x, out, B, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean_reduce(x, self.dim)