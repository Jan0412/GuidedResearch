import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    stride_b,
    stride_k,
    stride_d2,
    stride_b_out,
    stride_d2_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // dim2
    d2 = pid % dim2

    base_ptr = x_ptr + b * stride_b + d2 * stride_d2
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim1

    x_vals = tl.load(base_ptr + offsets * stride_k, mask=mask, other=0.0)
    sum_vals = tl.sum(x_vals)
    mean_val = sum_vals / dim1

    out_ptr_offset = out_ptr + b * stride_b_out + d2 * stride_d2_out
    tl.store(out_ptr_offset, mean_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    assert dim == 1, "Optimized kernel supports dim=1 only"

    batch_size, dim1, dim2 = x.shape
    out = torch.empty((batch_size, dim2), dtype=x.dtype, device=x.device)

    stride_b = x.stride(0)
    stride_k = x.stride(1)
    stride_d2 = x.stride(2)
    stride_b_out = out.stride(0)
    stride_d2_out = out.stride(1)

    BLOCK_SIZE = 4096

    grid = (batch_size * dim2,)
    mean_kernel[grid](
        x, out,
        batch_size, dim1, dim2,
        stride_b, stride_k, stride_d2,
        stride_b_out, stride_d2_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return triton_mean(x, self.dim)
        else:
            return torch.mean(x, dim=self.dim)