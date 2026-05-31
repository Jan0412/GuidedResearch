import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit(num_warps=4)
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < batch_size
    
    col_offsets = tl.arange(0, COL_BLOCK_SIZE)
    col_mask = col_offsets < dim
    
    mask = row_mask[:, None] & col_mask[None, :]
    
    row_sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute sum of squares along columns
    for col_start in range(0, dim, COL_BLOCK_SIZE):
        offsets = row_offsets[:, None] * dim + col_start + col_offsets[None, :]
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        row_sum_sq += tl.sum(x_block * x_block, axis=1)
        
    # Compute inverse square root with epsilon for numerical stability
    inv_sqrt = tl.rsqrt(row_sum_sq + 1e-8)
    inv_sqrt = inv_sqrt[:, None]
    
    # Second pass: normalize and store results
    for col_start in range(0, dim, COL_BLOCK_SIZE):
        offsets = row_offsets[:, None] * dim + col_start + col_offsets[None, :]
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out_block = x_block * inv_sqrt
        tl.store(out_ptr + offsets, out_block, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 128
    COL_BLOCK_SIZE = 1024
    
    grid = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    l2_norm_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE, COL_BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)