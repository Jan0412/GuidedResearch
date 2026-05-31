import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduce_kernel(
    x_ptr, out_ptr,
    batch_size, dim1, dim2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // dim2
    idx2 = pid % dim2
    
    offsets = batch * dim1 * dim2 + tl.arange(0, BLOCK_SIZE) * dim2 + idx2
    mask = tl.arange(0, BLOCK_SIZE) < dim1
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    
    min_val = tl.min(x)
    tl.store(out_ptr + pid, min_val)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    if dim == 1:
        out_shape = (x.shape[0], x.shape[2])
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        grid = lambda meta: (out_shape[0] * out_shape[1],)
        min_reduce_kernel[grid](x, out, x.shape[0], x.shape[1], x.shape[2], BLOCK_SIZE=4096)
        return out
    elif dim == 2:
        x_t = x.transpose(1, 2).contiguous()
        out_t = torch.empty((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        grid = lambda meta: (x.shape[0] * x.shape[1],)
        min_reduce_kernel[grid](x_t, out_t, x.shape[0], x.shape[2], x.shape[1], BLOCK_SIZE=4096)
        return out_t.transpose(1, 2)
    else:
        raise ValueError("Only dim=1 and dim=2 are supported.")


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min(x, self.dim)