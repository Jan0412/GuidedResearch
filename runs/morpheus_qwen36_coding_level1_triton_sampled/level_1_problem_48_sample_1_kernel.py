import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // dim2
    j = pid % dim2

    base_offset = b * dim1 * dim2 + j
    stride = dim2

    sum_val = 0.0
    num_blocks = (dim1 + BLOCK_SIZE - 1) // BLOCK_SIZE

    for k in range(num_blocks):
        i_offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = i_offsets < dim1
        offsets = base_offset + i_offsets * stride
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_block)

    out_val = sum_val / dim1
    tl.store(out_ptr + pid, out_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    assert dim == 1, "Optimized kernel supports dim=1 only"
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    out = torch.empty((batch_size, dim2), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    grid = (batch_size * dim2,)
    mean_kernel[grid](x, out, dim1, dim2, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return triton_mean(x, self.dim)
        return torch.mean(x, dim=self.dim)