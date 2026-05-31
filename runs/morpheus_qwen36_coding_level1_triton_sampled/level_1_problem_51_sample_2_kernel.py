import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // dim2
    last_dim_idx = pid % dim2
    
    base_ptr = x_ptr + batch_idx * dim1 * dim2 + last_dim_idx
    
    max_val = -float('inf')
    max_idx = 0
    
    for start in range(0, dim1, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        x_vals = tl.load(base_ptr + offsets * dim2, mask=mask, other=-float('inf'))
        
        local_max = tl.max(x_vals, axis=0)
        local_idx = tl.argmax(x_vals, axis=0)
        
        if local_max > max_val:
            max_val = local_max
            max_idx = start + local_idx
            
    tl.store(out_ptr + pid, max_idx.to(tl.int64))


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim1, dim2 = x.shape
    assert dim == 1, "Kernel currently optimized for dim=1."
    
    out = torch.empty(batch_size, dim2, dtype=torch.int64, device=x.device)
    
    num_programs = batch_size * dim2
    BLOCK_SIZE = 128
    
    argmax_kernel[(num_programs,)](
        x, out, batch_size, dim1, dim2, BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)