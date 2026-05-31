import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    min_val = float('inf')
    min_idx = -1
    
    num_chunks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(num_chunks):
        offsets = row_idx * row_stride + chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_idx * row_stride + n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
        
        chunk_min_val = tl.min(x, axis=0)
        chunk_min_idx = tl.argmin(x, axis=0) + chunk * BLOCK_SIZE
        
        if chunk_min_val < min_val:
            min_val = chunk_min_val
            min_idx = chunk_min_idx
            
    tl.store(out_ptr + row_idx, min_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    if dim == 1:
        batch_size = x.shape[0]
        row_size = x.shape[1]
        out = torch.empty(batch_size, dtype=torch.int32, device=x.device)
        
        BLOCK_SIZE = 128
        grid = (batch_size,)
        
        argmin_kernel[grid](x, out, row_size, row_size, BLOCK_SIZE=BLOCK_SIZE)
        return out
    else:
        return torch.argmin(x, dim=dim)


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)