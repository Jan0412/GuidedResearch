import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    X,
    Out,
    N,  # Number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    # Row index
    row_start = tl.program_id(0)
    
    # Offset to the start of the row
    row_offset = row_start * N
    
    # Compute max for numerical stability (first pass)
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_offset + cols, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        row_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, row_max)
    
    # Compute exp(x - max) and sum (second pass)
    sum_val = tl.full([1], 0, dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        exp_x = tl.exp(x - max_val)
        tl.store(Out + row_offset + cols, exp_x, mask=mask)
        sum_val += tl.sum(exp_x, 0)
    
    # Normalize by sum (third pass)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        out = tl.load(Out + row_offset + cols, mask=mask)
        out = out / sum_val
        tl.store(Out + row_offset + cols, out, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton-based softmax implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    n_cols = shape[dim]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Reshape to 2D for easier processing
    x_2d = x.view(n_rows, n_cols)
    out_2d = out.view(n_rows, n_cols)
    
    # Determine block size
    BLOCK_SIZE = 256
    
    # Calculate grid
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](x_2d, out_2d, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-based Softmax activation to the input tensor.
        """
        return triton_softmax(x, dim=1)