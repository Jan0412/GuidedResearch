import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    seq_len,
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * seq_len
    out_ptr = out_ptr + row_idx * seq_len
    
    current_sum = 0.0
    for i in range(seq_len):
        val = tl.load(row_ptr + i)
        current_sum += val
        tl.store(out_ptr + i, current_sum)


def triton_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "Input must be a 2D CUDA tensor"
    assert dim == 1, "Currently only supports dim=1"
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, seq_len = x.shape
    
    grid = (batch_size,)
    
    cumsum_kernel[grid](x, out, seq_len)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)


def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return [1]