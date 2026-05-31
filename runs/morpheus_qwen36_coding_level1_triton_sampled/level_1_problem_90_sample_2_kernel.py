import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    out = tl.associative_scan(x, 0, lambda a, b: a * b)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    # Move the target dimension to the last axis to ensure contiguous memory access
    x = x.transpose(dim, -1).contiguous()
    out = torch.empty_like(x)
    
    dim_size = x.shape[-1]
    batch_size = x.numel() // dim_size
    
    grid = lambda meta: (batch_size,)
    
    cumprod_kernel[grid](x, out, dim_size, BLOCK_SIZE=dim_size)
    
    # Transpose the output back to the original dimension order
    return out.transpose(-1, dim)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)