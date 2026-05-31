import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * n_elements_dim
    
    num_blocks = (n_elements_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block in range(num_blocks):
        offsets = base_offset + block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (base_offset + n_elements_dim)
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = tl.cumsum(x) - x
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_cumsum_last_dim(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous(), "Input must be contiguous CUDA tensor."
    
    n_elements = x.numel()
    n_elements_dim = x.shape[-1]
    n_slices = n_elements // n_elements_dim
    
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    
    grid = (n_slices,)
    
    cumsum_kernel[grid](x, out, n_elements_dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim == x.dim() - 1:
            return triton_cumsum_last_dim(x)
        else:
            x_moved = x.movedim(self.dim, -1).contiguous()
            out_moved = triton_cumsum_last_dim(x_moved)
            return out_moved.movedim(-1, self.dim).contiguous()