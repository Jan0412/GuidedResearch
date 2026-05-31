import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for i in range(0, dim_size, BLOCK_SIZE):
        block_offsets = i + offsets
        mask = block_offsets < dim_size
        x = tl.load(x_ptr + pid * dim_size + block_offsets, mask=mask, other=0.0)
        acc = tl.reduce(x, axis=0) + acc
    out = acc / dim_size
    tl.store(out_ptr + pid, out)

def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    # Permute so that the reduction dimension becomes the last dimension
    if dim != x.dim() - 1:
        new_dims = list(range(x.dim()))
        new_dims.remove(dim)
        new_dims.append(dim)
        x = x.permute(new_dims)
        dim = x.dim() - 1
    
    dim_size = x.size(dim)
    n_elements = x.numel() // dim_size
    
    out_shape = list(x.shape)
    out_shape.pop(dim)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    grid = (n_elements,)
    
    mean_kernel[grid](x, out, n_elements, dim_size, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)