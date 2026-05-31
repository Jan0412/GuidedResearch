import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr, y_ptr, stride_x, stride_y, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    for row in tl.program_id(0):
        x_ptr += row * stride_x
        y_ptr += row * stride_y
        
        prev_sum = 0.0
        for col in range(0, n_cols, BLOCK_SIZE):
            offsets = col + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            p_block = tl.associative_scan(x_block, axis=0, op="add")
            p_block += prev_sum
            
            tl.store(y_ptr + offsets, p_block, mask=mask)
            
            prev_sum += tl.sum(x_block)


def triton_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    if dim == 0:
        x = x.t().contiguous()
        out = triton_cumsum(x, dim=1)
        return out.t()
    
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    
    grid = (n_rows,)
    cumsum_kernel[grid](x, out, x.stride(0), x.stride(1), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)