import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_f_ptr, out_ptr,
    dim_size,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    row_start = pid_row * dim_size
    block_start = pid_block * BLOCK_SIZE
    
    offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
    mask_offsets = offsets < dim_size
    
    x = tl.load(x_ptr + offsets, mask=mask_offsets, other=0.0)
    m = tl.load(mask_f_ptr + offsets, mask=mask_offsets, other=0.0)
    
    z = x * m
    
    final_acc, out_vals = tl.scan(lambda acc, val: (acc + val, acc + val), 0.0, z)
    
    tl.store(out_ptr + offsets, out_vals, mask=mask_offsets)


def triton_masked_cumsum(x, mask, dim):
    assert x.is_cuda and mask.is_cuda
    assert x.dtype == torch.float32
    assert dim == x.ndim - 1, "Only support cumsum on last dimension"
    
    mask_f = mask.float()
    out = torch.empty_like(x)
    dim_size = x.shape[dim]
    num_rows = x.numel() // dim_size
    
    BLOCK_SIZE = 1024
    
    grid = (num_rows, triton.cdiv(dim_size, BLOCK_SIZE))
    
    masked_cumsum_kernel[grid](x, mask_f, out, dim_size, num_rows, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)