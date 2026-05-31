import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr,
    out_ptr,
    M, N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    mask_row = row_idx < M
    
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    
    for start_n in range(0, N, BLOCK_SIZE_N):
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = n_offsets < N
        
        x = tl.load(x_ptr + row_idx * N + n_offsets, mask=mask_n, other=-float('inf'))
        local_max = tl.max(x)
        max_val = tl.maximum(max_val, local_max)
        
    tl.store(out_ptr + row_idx, max_val, mask=mask_row)


def triton_max_reduce(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 3
    x = x.contiguous()
    M, _, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_N = 128
    
    grid = (M,)
    max_reduce_kernel[grid](x, out, M, N, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 2:
            return triton_max_reduce(x)
        elif self.dim == 1:
            x_t = x.transpose(1, 2)
            return triton_max_reduce(x_t)
        elif self.dim == 0:
            x_t = x.permute(2, 1, 0)
            return triton_max_reduce(x_t).transpose(0, 1)
        else:
            raise ValueError("Unsupported dim")