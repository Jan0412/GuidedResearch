import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr, out_ptr, n_rows, row_len, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_len
    
    num_blocks = (row_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cum_prod = 1.0
    
    for b in range(num_blocks):
        block_start = b * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=1.0)
        
        # Compute local prefix product sequentially
        local_out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        acc = 1.0
        for i in range(BLOCK_SIZE):
            acc *= x[i]
            local_out[i] = acc
            
        # Scale by cumulative product of previous blocks
        local_out *= block_cum_prod
        
        # Store result
        tl.store(out_ptr + row_start + offsets, local_out, mask=mask)
        
        # Update block_cum_prod for the next block
        block_cum_prod = local_out[-1]


def triton_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    needs_move = dim != x.dim() - 1
    if needs_move:
        x = x.movedim(dim, -1)
        
    n_rows = x.numel() // x.shape[-1]
    row_len = x.shape[-1]
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    cumprod_kernel[grid](x, out, n_rows, row_len, BLOCK_SIZE=BLOCK_SIZE)
    
    if needs_move:
        out = out.movedim(-1, dim)
        
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)