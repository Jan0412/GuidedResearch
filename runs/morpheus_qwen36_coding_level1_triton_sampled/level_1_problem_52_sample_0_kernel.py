import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr, out_ptr, batch_size, dim1, dim2,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // dim2
    d2 = pid % dim2

    # Base offset for the current (b, d2) slice
    base_offset = b * dim1 * dim2 + d2

    min_val = float('inf')
    min_idx = 0

    # Iterate over dim1 in blocks
    for i in range(0, dim1, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        # Strided access along dim1 (stride is dim2)
        x_vals = tl.load(x_ptr + base_offset + offsets * dim2, mask=mask, other=float('inf'))
        
        local_min_val = tl.min(x_vals)
        local_min_idx = tl.argmin(x_vals)

        # Update global minimum, preserving first occurrence
        if local_min_val < min_val:
            min_val = local_min_val
            min_idx = i + local_min_idx

    tl.store(out_ptr + pid, min_idx)


def triton_argmin(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    
    out = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)
    
    BLOCK_SIZE = 128
    grid = (batch_size * dim2,)
    argmin_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return triton_argmin(x, self.dim)
        # Fallback for other dimensions
        return torch.argmin(x, dim=self.dim)


batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [dim]