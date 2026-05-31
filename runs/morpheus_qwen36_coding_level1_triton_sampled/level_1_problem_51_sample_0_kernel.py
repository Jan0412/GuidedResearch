import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr, out_ptr, B, D1, D2, BLOCK_SIZE: tl.constexpr
):
    b = tl.program_id(0)
    d2 = tl.program_id(1)
    
    x_offset_base = b * D1 * D2 + d2
    
    max_val = -float('inf')
    max_idx = 0
    
    num_blocks = (D1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_id in range(num_blocks):
        start = block_id * BLOCK_SIZE
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < D1
        
        vals = tl.load(x_ptr + x_offset_base + off * D2, mask=mask, other=-float('inf'))
        
        local_max = tl.max(vals, axis=0)
        local_idx = tl.argmax(vals, axis=0)
        
        local_idx = tl.where(mask, local_idx + start, 0)
        local_max = tl.where(mask, local_max, -float('inf'))
        
        new_max = tl.where(local_max > max_val, local_max, max_val)
        new_idx = tl.where(local_max > max_val, local_idx, max_idx)
        
        max_val = new_max
        max_idx = new_idx
        
    tl.store(out_ptr + b * D2 + d2, max_idx)


def triton_argmax(x: torch.Tensor, dim: int):
    assert x.is_cuda
    x = x.contiguous()
    B, D1, D2 = x.shape
    assert dim == 1, "Only dim=1 supported for this kernel"
    
    out = torch.empty((B, D2), dtype=torch.int32, device=x.device)
    
    BLOCK_SIZE = 128
    grid = (B, D2)
    
    argmax_kernel[grid](x, out, B, D1, D2, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)