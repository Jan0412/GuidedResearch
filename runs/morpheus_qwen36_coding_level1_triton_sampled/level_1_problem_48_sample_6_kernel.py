import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(x_ptr, out_ptr, x_shape_ptr, x_strides_ptr, n_elements, dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute base offset for pid in non-reduction space
    offset = 0
    temp = pid
    for d in range(3):
        if d != dim:
            stride = tl.load(x_strides_ptr + d)
            shape_d = tl.load(x_shape_ptr + d)
            idx = temp % shape_d
            temp //= shape_d
            offset += idx * stride
            
    # Reduce over the specified dimension
    sum_val = 0.0
    n_reduce = tl.load(x_shape_ptr + dim)
    for i in range(0, n_reduce, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE) + i * tl.load(x_strides_ptr + dim)
        mask = block_offsets < n_elements
        x_vals = tl.load(x_ptr + block_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals)
        
    out_val = sum_val / n_reduce
    tl.store(out_ptr + pid, out_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    n_elements = x.numel()
    n_out = n_elements // x.shape[dim]
    out = torch.empty(n_out, dtype=x.dtype, device=x.device)
    
    x_shape = torch.tensor(x.shape, dtype=torch.int32, device=x.device)
    x_strides = torch.tensor(x.stride(), dtype=torch.int32, device=x.device)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (n_out,)
    
    mean_kernel[grid](x, out, x_shape, x_strides, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)