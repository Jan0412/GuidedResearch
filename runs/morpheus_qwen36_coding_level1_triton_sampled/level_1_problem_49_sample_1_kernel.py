import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr,
    out_ptr,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    
    max_val = -float('inf')
    
    for start in range(0, dim2, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim2
        x = tl.load(x_ptr + row_idx * dim2 + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
        
    tl.store(out_ptr + row_idx, max_val)


def triton_max_reduce(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "Input must be a 2D CUDA tensor."
    x = x.contiguous()
    dim1, dim2 = x.shape
    out = torch.empty(dim1, dtype=x.dtype, device=x.device)
    grid = lambda meta: (dim1,)
    max_reduce_kernel[grid](x, out, dim1, dim2, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move the reduction dimension to the last position for efficient kernel execution
        x_moved = x.movedim(self.dim, -1).contiguous()
        return triton_max_reduce(x_moved)