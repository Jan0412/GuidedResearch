import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_size

    # First pass: compute row-wise maximum
    M = -float('inf')
    for block_start in range(0, row_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        x_max = tl.max(x, axis=0)
        M = tl.maximum(M, x_max)

    # Second pass: compute sum of exponentials
    S = 0.0
    for block_start in range(0, row_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        S += tl.sum(tl.exp(x - M), axis=0)

    # Third pass: compute softmax and store
    for block_start in range(0, row_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        out = tl.exp(x - M) / S
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    n_rows, row_size = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024  # Tunable block size
    
    grid = (n_rows,)
    softmax_kernel[grid](x, out, n_rows, row_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []