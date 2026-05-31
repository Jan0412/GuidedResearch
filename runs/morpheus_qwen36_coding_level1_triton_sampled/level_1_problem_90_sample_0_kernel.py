import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * stride_x_row
    
    num_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    
    prefix_prod = 1.0
    
    for block_id in range(num_blocks):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        x_block = tl.load(x_ptr + row_offset + offsets, mask=mask, other=1.0)
        
        current_prod = 1.0
        
        for i in range(BLOCK_SIZE):
            current_prod *= x_block[i]
            out_val = prefix_prod * current_prod
            
            tl.store(out_ptr + row_offset + block_id * BLOCK_SIZE + i, out_val, mask=mask[i])
            
        prefix_prod = current_prod


def triton_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    original_dim = dim
    if dim == 0:
        x = x.t()
        dim = 1
    
    batch_size, n_cols = x.shape
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    
    cumprod_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if original_dim == 0:
        out = out.t()
        
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)