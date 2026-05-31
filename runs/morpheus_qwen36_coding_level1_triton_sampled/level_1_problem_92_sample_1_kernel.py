import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * num_cols
    out_row_ptr = out_ptr + row_idx * num_cols

    running_sum = 0.0
    num_blocks = tl.cdiv(num_cols, BLOCK_SIZE)

    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols

        x_block = tl.load(row_ptr + offsets, mask=mask, other=0.0)

        # Compute prefix sum for the block
        prefix_sum = 0.0
        for i in range(BLOCK_SIZE):
            if block_start + i < num_cols:
                prefix_sum += x_block[i]
                x_block[i] = prefix_sum
            else:
                x_block[i] = 0.0

        # Compute exclusive cumsum for the block
        out_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(BLOCK_SIZE):
            if block_start + i < num_cols:
                if i == 0:
                    out_block[i] = running_sum
                else:
                    out_block[i] = running_sum + x_block[i - 1]
            else:
                out_block[i] = 0.0

        tl.store(out_row_ptr + offsets, out_block, mask=mask)

        # Update running sum
        running_sum += tl.sum(x_block)


def triton_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert dim == 1, "Only dim=1 is supported in this implementation."

    x = x.contiguous()
    num_rows, num_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 128

    grid = (num_rows,)
    exclusive_cumsum_kernel[grid](x, out, num_rows, num_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_exclusive_cumsum(x, self.dim)