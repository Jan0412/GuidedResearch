import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE_X: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    x_offset = pid * dim

    # First pass: compute sum of absolute values
    sum_abs = 0.0
    for start in range(0, dim, BLOCK_SIZE_X):
        offsets = start + tl.arange(0, BLOCK_SIZE_X)
        mask = offsets < dim
        x_vals = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(x_vals)
        sum_abs = tl.sum(abs_vals)

    mean_abs = sum_abs / dim

    # Second pass: divide original values by mean and store result
    for start in range(0, dim, BLOCK_SIZE_X):
        offsets = start + tl.arange(0, BLOCK_SIZE_X)
        mask = offsets < dim
        x_vals = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        out_vals = x_vals / mean_abs
        tl.store(out_ptr + x_offset + offsets, out_vals, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    batch_size, dim = x.shape
    BLOCK_SIZE_X = 256  # Tunable block size for dimension reduction

    grid = (batch_size,)
    l1_norm_kernel[grid](
        x, out, batch_size, dim, BLOCK_SIZE_X=BLOCK_SIZE_X
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)