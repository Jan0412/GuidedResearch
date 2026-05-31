import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, out_ptr, batch_size, dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * dim
    
    max_val = -float('inf')
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
        
    sum_exp = 0.0
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
        
    log_sum_exp = tl.log(sum_exp)
    
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        out = x - max_val - log_sum_exp
        tl.store(out_ptr + row_offset + offsets, out, mask=mask)

def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 2048
    grid = (batch_size,)
    log_softmax_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)