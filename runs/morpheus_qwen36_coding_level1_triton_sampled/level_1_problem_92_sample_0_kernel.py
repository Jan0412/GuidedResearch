import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr, out_ptr, seq_len, stride_row, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_row
    out_ptr += row_idx * stride_row
    
    running_sum = 0.0
    for start in range(0, seq_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        inc_sum = tl.cumsum(x)
        out = inc_sum - x + running_sum
        
        tl.store(out_ptr + offsets, out, mask=mask)
        running_sum += tl.sum(x)


def triton_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    x_moved = x.movedim(dim, -1)
    orig_shape = x_moved.shape
    seq_len = orig_shape[-1]
    num_rows = orig_shape.numel() // seq_len
    
    out_moved = torch.empty_like(x_moved)
    stride_row = x_moved.stride(-2) if len(orig_shape) > 1 else 1
    
    BLOCK_SIZE = 128
    grid = (num_rows,)
    exclusive_cumsum_kernel[grid](x_moved, out_moved, seq_len, stride_row, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out_moved.movedim(-1, dim)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)