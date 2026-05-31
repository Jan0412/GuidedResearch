import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * stride_x
    
    m = -float('inf')
    l = 0.0
    
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        x_block = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        
        m_i = tl.max(x_block, axis=0)
        l_i = tl.sum(tl.exp(x_block - m_i), axis=0)
        
        m_new = tl.maximum(m, m_i)
        l = tl.exp(m - m_new) * l + tl.exp(m_i - m_new) * l_i
        m = m_new
        
    log_sum_exp = tl.log(l) + m
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        out_block = x_block - log_sum_exp
        tl.store(out_ptr + row_offset + offsets, out_block, mask=mask)


def triton_log_softmax(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, dim = x.shape
    stride_x = dim
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    log_softmax_kernel[grid](x, out, stride_x, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)