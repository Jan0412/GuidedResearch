import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_ptr, out_ptr,
    row_len,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * row_len
    mask_row_ptr = mask_ptr + row_idx * row_len
    out_row_ptr = out_ptr + row_idx * row_len
    
    acc = 0.0
    for start in range(0, row_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        m_vals = tl.load(mask_row_ptr + offsets, mask=mask, other=0)
        vals = tl.where(m_vals, x_vals, 0.0)
        
        for i in range(BLOCK_SIZE):
            acc += vals[i]
            tl.store(out_row_ptr + offsets[i], acc, mask=mask[i])

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    assert x.is_cuda and mask.is_cuda
    assert x.shape == mask.shape
    assert dim == 1, "This implementation is optimized for dim=1"
    
    x = x.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(x)
    
    batch_size = x.shape[0]
    row_len = x.shape[1]
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    masked_cumsum_kernel[grid](x, mask, out, row_len, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)