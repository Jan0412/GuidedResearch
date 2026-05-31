import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr, out_ptr, stride_x, stride_out,
    dim_size, batch_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid
    if row_idx >= batch_size:
        return
    
    num_tiles = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    acc = 0.0
    for tile in range(num_tiles):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        x = tl.load(x_ptr + row_idx * stride_x + offsets, mask=mask, other=0.0)
        # Compute reverse cumulative sum within the tile
        tile_out = tl.associative_scan(x, reverse=True)
        # Add the accumulated sum from previous tiles
        tile_out = tile_out + acc
        tl.store(out_ptr + row_idx * stride_out + offsets, tile_out, mask=mask)
        # Update accumulator for the next tile
        acc = tl.sum(x) + acc


def triton_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, dim_size = x.shape
    stride_x = x.stride(0)
    stride_out = out.stride(0)
    
    BLOCK_SIZE = 256
    grid = lambda meta: (batch_size,)
    
    reverse_cumsum_kernel[grid](x, out, stride_x, stride_out, dim_size, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)