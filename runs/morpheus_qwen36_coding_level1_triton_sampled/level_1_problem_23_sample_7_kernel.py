import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # First pass: find max per row for numerical stability
    row_max = -float('inf')
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offset = start_col + cols
        mask = offset < n_cols
        x_chunk = tl.load(x_ptr + row_idx * n_cols + offset, mask=mask, other=-float('inf'))
        chunk_max = tl.max(x_chunk, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Second pass: compute exp, accumulate sum, and store
    row_sum = 0.0
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offset = start_col + cols
        mask = offset < n_cols
        x_chunk = tl.load(x_ptr + row_idx * n_cols + offset, mask=mask, other=-float('inf'))
        exp_chunk = tl.where(mask, tl.exp(x_chunk - row_max), 0.0)
        row_sum += tl.sum(exp_chunk, axis=0)
        out_chunk = tl.where(mask, exp_chunk / row_sum, 0.0)
        tl.store(out_ptr + row_idx * n_cols + offset, out_chunk, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the custom Triton softmax kernel.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # Tunable block size (power of 2)
    grid = (n_rows,)
    softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for Softmax.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)