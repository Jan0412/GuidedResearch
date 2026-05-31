import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * n_cols
    
    max_val = float('-inf')
    max_idx = 0
    
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        
        local_max = tl.max(x, axis=0)
        local_idx = tl.argmax(x, axis=0)
        
        max_idx = tl.select(local_max > max_val, local_idx + start_col, max_idx)
        max_val = tl.select(local_max > max_val, local_max, max_val)
        
    tl.store(out_ptr + row_idx, max_idx.to(tl.int32))


def triton_argmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    
    out = torch.empty(n_rows, dtype=torch.int32, device=x.device)
    
    BLOCK_SIZE = 128
    
    grid = (n_rows,)
    argmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != x.dim() - 1:
            x = x.permute(*range(self.dim), *range(self.dim + 1, x.dim()), self.dim)
        
        out = triton_argmax(x)
        return out