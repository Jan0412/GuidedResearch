import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * stride_x
    out_start = out_ptr + row_idx * stride_out
    
    # First pass: find the maximum value in the row for numerical stability
    max_val = -float('inf')
    for start in range(0, num_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        x = tl.load(row_start + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
        
    # Second pass: compute the sum of exponentials
    sum_exp = 0.0
    for start in range(0, num_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        x = tl.load(row_start + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
        
    # Third pass: normalize and store the result
    for start in range(0, num_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        x = tl.load(row_start + offsets, mask=mask, other=-float('inf'))
        tl.store(out_start + offsets, tl.exp(x - max_val) / sum_exp, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton softmax kernel.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    num_rows, num_cols = x.shape
    BLOCK_SIZE = 1024  # Tunable block size
    
    grid = (num_rows,)
    softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        num_rows, num_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)