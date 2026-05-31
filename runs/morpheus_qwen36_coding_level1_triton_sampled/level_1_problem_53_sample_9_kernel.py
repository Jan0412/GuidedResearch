import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_kernel(
    x_ptr,
    out_ptr,
    B,
    D1,
    D2,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if dim == 1:
        b = pid // D2
        d2 = pid % D2
        min_val = float('inf')
        for i in range(0, D1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < D1
            data = tl.load(x_ptr + b * D1 * D2 + offsets * D2 + d2, mask=mask, other=float('inf'))
            min_val = tl.minimum(min_val, tl.reduce(data, 0, tl.minimum))
        tl.store(out_ptr + b * D2 + d2, min_val)
    elif dim == 2:
        b = pid // D1
        d1 = pid % D1
        min_val = float('inf')
        for i in range(0, D2, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < D2
            data = tl.load(x_ptr + b * D1 * D2 + d1 * D2 + offsets, mask=mask, other=float('inf'))
            min_val = tl.minimum(min_val, tl.reduce(data, 0, tl.minimum))
        tl.store(out_ptr + b * D1 + d1, min_val)
    else:
        raise NotImplementedError("Only dim 1 and 2 are supported for optimization")


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, D1, D2 = x.shape
    
    if dim == 1:
        out_shape = (B, D2)
    elif dim == 2:
        out_shape = (B, D1)
    else:
        raise NotImplementedError("Only dim 1 and 2 are supported for optimization")
        
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_elements = out.numel()
    grid = (num_elements,)
    
    min_kernel[grid](x, out, B, D1, D2, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min(x, self.dim)