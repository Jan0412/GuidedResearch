import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols
    # Load chunk for the current row. Use inf for masked positions.
    x = tl.load(x_ptr + row_idx * num_cols + offsets, mask=mask, other=float('inf'))
    # Reduce along the column axis to find the minimum value
    min_val = tl.reduce(x, axis=0, fn=tl.minimum)
    tl.store(out_ptr + row_idx, min_val)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    # Ensure FP32 precision as requested
    x = x.to(torch.float32).contiguous()

    # Normalize dimension to handle negative indices
    dim = dim if dim >= 0 else x.ndim + dim

    # Move reduction dimension to the last axis for contiguous memory access
    x_moved = x.movedim(dim, -1)
    num_cols = x_moved.shape[-1]
    num_rows = x_moved.numel() // num_cols

    # Flatten all preceding dimensions into a single batch dimension
    x_flat = x_moved.reshape(-1, num_cols)

    # Prepare output tensor
    out_flat = torch.empty(num_rows, dtype=x.dtype, device=x.device)

    # Configure grid and block size
    BLOCK_SIZE = 128
    grid = (num_rows,)

    # Launch the Triton kernel
    min_kernel[grid](x_flat, out_flat, num_rows, num_cols, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape output back to the expected reduced shape
    return out_flat.reshape(x_moved.shape[:-1])


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min(x, self.dim)