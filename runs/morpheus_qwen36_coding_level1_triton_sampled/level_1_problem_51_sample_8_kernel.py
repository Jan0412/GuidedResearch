import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    
    base_offset = b * dim1 * dim2 + j
    x_ptr += base_offset
    
    max_val = -tl.inf
    max_idx = 0
    
    for i in range(0, dim1, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        vals = tl.load(x_ptr + offsets * dim2, mask=mask, other=-tl.inf)
        
        idx_in_block = tl.argmax(vals, axis=0)
        val_in_block = tl.max(vals, axis=0)
        
        if val_in_block > max_val:
            max_val = val_in_block
            max_idx = i + idx_in_block
            
    tl.store(out_ptr + b * dim2 + j, max_idx.to(tl.int64))

def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    out = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)
    
    BLOCK_SIZE = 64
    
    grid = (batch_size, dim2)
    argmax_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)