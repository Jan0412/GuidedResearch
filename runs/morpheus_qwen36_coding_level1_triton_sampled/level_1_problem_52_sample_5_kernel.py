import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    min_val = float('inf')
    min_idx = -1
    
    for i in range(0, dim1, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        x = tl.load(x_ptr + pid_b * dim1 * dim2 + offsets * dim2 + pid_c, mask=mask, other=float('inf'))
        
        block_min = tl.min(x, axis=0)
        block_min_idx = tl.argmin(x, axis=0)
        
        cond = block_min < min_val
        min_val = tl.where(cond, block_min, min_val)
        min_idx = tl.where(cond, i + block_min_idx, min_idx)
        
    tl.store(out_ptr + pid_b * dim2 + pid_c, min_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    assert dim == 1, "Currently only supports dim=1"
    out = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)
    
    BLOCK_SIZE = 128
    grid = (batch_size, dim2)
    
    argmin_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)