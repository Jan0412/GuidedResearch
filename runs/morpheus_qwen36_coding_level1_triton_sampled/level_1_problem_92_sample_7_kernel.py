import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_block_sums_kernel(x_ptr, block_sums_ptr, N, stride_row, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offset = pid * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x)
    tl.store(block_sums_ptr + pid, block_sum)


@triton.jit
def exclusive_cumsum_kernel(x_ptr, out_ptr, N, stride_row, block_prefix_sums_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offset = pid * stride_row
    prefix_sum = tl.load(block_prefix_sums_ptr + pid)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    x_scan = tl.associative_scan(x, tl.ops.add, exclusive=True)
    out = prefix_sum + x_scan
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


def triton_exclusive_cumsum(x):
    N = x.shape[-1]
    M = x.numel() // N
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_sums = torch.zeros(num_blocks, dtype=torch.float32, device=x.device)
    grid = (M,)
    compute_block_sums_kernel[grid](x, block_sums, N, x.stride(0), BLOCK_SIZE)
    block_prefix_sums = torch.cumsum(block_sums, dim=0)
    exclusive_cumsum_kernel[grid](x, out, N, x.stride(0), block_prefix_sums, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Transpose so that the cumsum dimension is last for efficient memory access
        x_t = x.transpose(self.dim, -1).contiguous()
        out_t = triton_exclusive_cumsum(x_t)
        out = out_t.transpose(-1, self.dim)
        return out