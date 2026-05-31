import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    batch_size, dim,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    row_offset = row_idx * stride_x_row
    col_offset = col_block_idx * BLOCK_SIZE
    cols = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = cols < dim
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Optimized for FP32 precision with fused multiply-add patterns
    out = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + row_offset + cols, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch_size, (dim + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    gelu_kernel[grid](x, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1), batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)