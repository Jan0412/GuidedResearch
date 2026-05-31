import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    input_ptr, output_ptr,
    n_elements, n_scan_dim,
    stride_input, stride_output,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * stride_input
    
    running_sum = 0.0
    
    for start in range(0, n_scan_dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_scan_dim
        x = tl.load(input_ptr + row_offset + offsets * stride_input, mask=mask, other=0.0)
        
        local_prefix = tl.associative_scan(x, axis=0)
        local_prefix += running_sum
        
        tl.store(output_ptr + row_offset + offsets * stride_input, local_prefix, mask=mask)
        
        running_sum += tl.sum(x, axis=0)


def triton_cumsum(x, dim):
    assert x.is_cuda
    x = x.contiguous()
    x = x.movedim(dim, -1)
    n_scan_dim = x.shape[-1]
    n_other_dims = x.numel() // n_scan_dim
    
    if n_scan_dim == 0 or x.numel() == 0:
        return x
    
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    
    grid = (n_other_dims,)
    
    cumsum_kernel[grid](
        x, out,
        x.numel(), n_scan_dim,
        1, 1,
        BLOCK_SIZE
    )
    
    out = out.movedim(-1, dim)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)