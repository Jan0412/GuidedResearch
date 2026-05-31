import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['N']
)
@triton.jit
def l2_norm_kernel(
    x_ptr, out_ptr, M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * N
    
    sum_sq = 0.0
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First pass: compute sum of squares
    for i in range(num_blocks):
        offsets = row_offset + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (row_offset + N)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
        
    inv_rms = 1.0 / tl.sqrt(sum_sq)
    
    # Second pass: normalize
    for i in range(num_blocks):
        offsets = row_offset + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (row_offset + N)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * inv_rms, mask=mask)

def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    M, N = x.shape
    
    grid = lambda meta: (M,)
    l2_norm_kernel[grid](x, out, M, N, BLOCK_SIZE=meta['BLOCK_SIZE'])
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)