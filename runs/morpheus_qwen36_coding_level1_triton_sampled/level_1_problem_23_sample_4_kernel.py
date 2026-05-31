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
    row_start = row_idx * n_cols
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load input row
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
    
    # Compute softmax along the row
    out = tl.softmax(x, axis=0)
    
    # Store output row
    tl.store(out_ptr + row_start + cols, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    grid = (n_rows,)
    
    softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)