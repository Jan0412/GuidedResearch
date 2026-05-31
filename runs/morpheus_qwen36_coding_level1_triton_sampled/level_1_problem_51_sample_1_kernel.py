import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr, out_ptr, batch_size, dim1, dim2,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // dim2
    d2_idx = pid % dim2
    
    max_val = -float('inf')
    max_idx = 0
    
    num_blocks = (dim1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_off in range(num_blocks):
        block_start = block_off * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        
        ptr_offset = batch_idx * dim1 * dim2 + block_start * dim2 + d2_idx
        x_vals = tl.load(x_ptr + ptr_offset + tl.arange(0, BLOCK_SIZE) * dim2, mask=mask, other=-float('inf'))
        
        local_max = tl.max(x_vals, axis=0)
        if local_max > max_val:
            max_val = local_max
            local_idx = tl.argmax(x_vals, axis=0)
            max_idx = block_start + local_idx
            
    tl.store(out_ptr + pid, max_idx)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    assert dim == 1, "Optimized for dim=1"
    
    out = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)
    
    BLOCK_SIZE = 128
    num_outputs = batch_size * dim2
    grid = (num_outputs,)
    
    argmax_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out.to(torch.int64)


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)