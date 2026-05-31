import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_ptr, out_ptr,
    batch_size, input_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < batch_size:
        row_x = x_ptr + pid * input_size
        row_mask = mask_ptr + pid * input_size
        row_out = out_ptr + pid * input_size
        
        acc = 0.0
        for i in range(0, input_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask_offsets = offsets < input_size
            x_vals = tl.load(row_x + offsets, mask=mask_offsets, other=0.0)
            m_vals = tl.load(row_mask + offsets, mask=mask_offsets, other=False)
            
            vals = x_vals * m_vals.to(tl.float32)
            psum = tl.cumsum(vals, axis=0)
            
            tl.store(row_out + offsets, acc + psum, mask=mask_offsets)
            acc += tl.sum(vals)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and mask.is_cuda
    assert x.dtype == torch.float32
    assert dim == 1
    
    x = x.contiguous()
    mask = mask.contiguous()
    
    batch_size, input_size = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (batch_size,)
    
    masked_cumsum_kernel[grid](x, mask, out, batch_size, input_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)