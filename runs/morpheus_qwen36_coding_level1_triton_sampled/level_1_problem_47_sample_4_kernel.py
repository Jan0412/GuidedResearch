import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    k_size,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // n_cols
    m = pid % n_cols
    
    base_offset = n * k_size * n_cols + m
    
    acc = 0.0
    num_blocks = k_size // BLOCK_SIZE_K
    for i in range(num_blocks):
        k_start = i * BLOCK_SIZE_K
        offsets = base_offset + tl.arange(0, BLOCK_SIZE_K) * n_cols
        mask = (k_start + tl.arange(0, BLOCK_SIZE_K)) < k_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x)
        
    tl.store(out_ptr + pid, acc)


def triton_sum_reduce(x: torch.Tensor, dim: int):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    out_shape = list(x.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    n_rows = x.shape[0]
    n_cols = x.shape[2]
    k_size = x.shape[1]
    BLOCK_SIZE_K = 128
    
    grid = (n_rows * n_cols,)
    sum_reduce_kernel[grid](x, out, n_rows, n_cols, k_size, BLOCK_SIZE_K=BLOCK_SIZE_K)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)