import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    D1,
    D2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // D2
    c = pid % D2
    
    max_val = -float('inf')
    max_idx = 0
    
    num_blocks = (D1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for k in range(num_blocks):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D1
        
        ptr = x_ptr + b * D1 * D2 + offsets * D2 + c
        vals = tl.load(ptr, mask=mask, other=-float('inf'))
        
        local_idx = tl.argmax(vals, axis=0)
        local_val = tl.max(vals, axis=0)
        
        if local_val > max_val:
            max_val = local_val
            max_idx = k * BLOCK_SIZE + local_idx
            
    tl.store(out_ptr + b * D2 + c, max_idx)


def triton_argmax(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    B, D1, D2 = x.shape
    assert dim == 1, "Custom kernel optimized for dim=1."
    
    out = torch.empty((B, D2), dtype=torch.int32, device=x.device)
    BLOCK_SIZE = 128
    num_programs = B * D2
    grid = (num_programs,)
    
    argmax_kernel[grid](x, out, D1, D2, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)