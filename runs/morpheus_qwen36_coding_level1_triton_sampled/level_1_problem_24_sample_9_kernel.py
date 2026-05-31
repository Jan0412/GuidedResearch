import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # First pass: compute max
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(x_ptr + row_start + block_offsets, mask=block_mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_block, axis=0))

    # Second pass: compute sum
    sum_val = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(x_ptr + row_start + block_offsets, mask=block_mask, other=0.0)
        exp_x = tl.exp(x_block - max_val)
        sum_val += tl.sum(exp_x, axis=0)

    # Third pass: compute output
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(x_ptr + row_start + block_offsets, mask=block_mask, other=0.0)
        out_block = x_block - max_val - tl.log(sum_val)
        tl.store(out_ptr + row_start + block_offsets, out_block, mask=block_mask)


def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # Tunable parameter

    grid = (n_rows,)
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a LogSoftmax activation using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_log_softmax(x)