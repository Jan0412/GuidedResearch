import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    x_ptr, out_ptr, N, K,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    base_offset = pid * K
    offsets_k = tl.arange(0, BLOCK_K)
    mask = offsets_k < K
    
    x = tl.load(x_ptr + base_offset + offsets_k, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    tl.store(out_ptr + pid, max_val)

def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    x = x.contiguous()
    
    N = x.shape[:-1].numel()
    K = x.shape[-1]
    
    out = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
    
    BLOCK_K = 256
    grid = (N,)
    
    max_kernel[grid](x, out, N, K, BLOCK_K=BLOCK_K)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max(x, self.dim)