import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    B, D1, D2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    
    min_val = tl.full((1,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((1,), 0, dtype=tl.int32)
    
    for i in range(0, D1, BLOCK_SIZE):
        offsets_i = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets_i < D1
        
        ptr = x_ptr + b * D1 * D2 + j + offsets_i * D2
        vals = tl.load(ptr, mask=mask, other=float('inf'))
        
        current_val = tl.full((1,), float('inf'), dtype=tl.float32)
        current_idx = tl.full((1,), 0, dtype=tl.int32)
        
        for k in range(BLOCK_SIZE):
            v = vals[k]
            m = v < current_val
            current_val = tl.where(m, v, current_val)
            current_idx = tl.where(m, k + i, current_idx)
            
        m_update = current_val < min_val
        min_val = tl.where(m_update, current_val, min_val)
        min_idx = tl.where(m_update, current_idx, min_idx)
        
    tl.store(out_ptr + b * D2 + j, min_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    B, D1, D2 = x.shape
    out = torch.empty((B, D2), dtype=torch.int32, device=x.device)
    
    BLOCK_SIZE = 64
    grid = (B, D2)
    
    argmin_kernel[grid](x, out, B, D1, D2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)