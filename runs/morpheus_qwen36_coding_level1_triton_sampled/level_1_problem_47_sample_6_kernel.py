import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr, out_ptr,
    B, D1, D2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = B * D2
    
    if pid < num_blocks:
        b = pid // D2
        d2 = pid % D2
        
        base_ptr = b * D1 * D2 + d2
        out_ptr_idx = b * D2 + d2
        
        acc_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for start in range(0, D1, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < D1
            vals = tl.load(x_ptr + base_ptr + offsets * D2, mask=mask, other=0.0)
            acc_vec += vals
            
        acc = tl.sum(acc_vec)
        tl.store(out_ptr + out_ptr_idx, acc)


def triton_sum_reduce(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    B, D1, D2 = x.shape
    assert dim == 1, "This kernel is optimized for reduction over dim=1."
    
    out = torch.empty((B, 1, D2), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (B * D2,)
    
    sum_reduce_kernel[grid](x, out, B, D1, D2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)