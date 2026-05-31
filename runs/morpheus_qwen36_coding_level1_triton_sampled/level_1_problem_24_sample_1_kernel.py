import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * n_cols
    out_ptr += row_idx * n_cols
    
    # Pass 1: Find max value in the row for numerical stability
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_block, axis=0))
        
    # Pass 2: Compute sum of exp(x - max)
    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        exp_block = tl.exp(x_block - max_val)
        sum_exp += tl.sum(exp_block)
        
    log_sum_exp = tl.log(sum_exp)
    
    # Pass 3: Compute output: log_softmax(x) = x - max - log(sum(exp(x - max)))
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        out_block = x_block - max_val - log_sum_exp
        tl.store(out_ptr + offsets, out_block, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert dim == 1, "Currently only supports dim=1."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = 1024  # Tunable block size
    
    grid = (n_rows,)
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x, dim=self.dim)