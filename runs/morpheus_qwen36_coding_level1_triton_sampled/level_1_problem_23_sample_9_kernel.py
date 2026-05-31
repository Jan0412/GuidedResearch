import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, dim, stride_batch, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_batch
    out_ptr += row_idx * stride_batch
    
    max_val = -float('inf')
    sum_val = 0.0
    
    # Pass 1: Find max for numerical stability
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)
        
    # Pass 2: Compute sum of exp(x - max)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - max_val)
        sum_val += tl.sum(y, axis=0)
        
    # Pass 3: Compute softmax output
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - max_val)
        out = y / sum_val
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    stride_batch = x.stride(0)
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch_size,)
    softmax_kernel[grid](x, out, dim, stride_batch, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)