import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr, out_ptr, dim, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_ptr = x_ptr + row_idx * dim
    out_ptr = out_ptr + row_idx * dim

    # 1. Find max along the row for numerical stability
    max_val = float('-inf')
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(base_ptr + offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(x_block, axis=0))

    # 2. Compute exp(x - max) and accumulate sum
    sum_exp = 0.0
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(base_ptr + offsets, mask=mask, other=float('-inf'))
        x_shifted = x_block - max_val
        exp_x = tl.exp(x_shifted)
        sum_exp = sum_exp + tl.sum(exp_x, axis=0)

    # 3. Compute log(sum) and final output: log_softmax(x) = (x - max) - log(sum(exp(x - max)))
    log_sum = tl.log(sum_exp)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(base_ptr + offsets, mask=mask, other=float('-inf'))
        x_shifted = x_block - max_val
        out = x_shifted - log_sum
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    x = x.to(torch.float32)
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    log_softmax_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)


def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []