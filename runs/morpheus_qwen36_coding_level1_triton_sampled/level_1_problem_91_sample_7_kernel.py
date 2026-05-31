import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr, out_ptr, dim_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * dim_size
    x_ptr += row_offset
    out_ptr += row_offset
    
    running_sum = 0.0
    num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process blocks in reverse order to compute reverse cumulative sum in a single pass
    for block_idx in range(num_blocks - 1, -1, -1):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        running_sum = running_sum + x
        tl.store(out_ptr + offsets, running_sum, mask=mask)


def reverse_cumsum_triton(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size = x.shape[0]
    dim_size = x.shape[dim]
    BLOCK_SIZE = 128  # Tunable block size for memory coalescing
    
    grid = (batch_size,)
    reverse_cumsum_kernel[grid](x, out, dim_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return reverse_cumsum_triton(x, self.dim)


def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]


def get_init_inputs():
    return [dim]