import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    x_strides,
    x_shape,
    dim,
    dim_size,
    num_elements_out,
    BLOCK_SIZE_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_elements_out:
        return

    # Compute base offset for this output element
    offset = 0
    temp = pid

    # Unroll for 3 dimensions
    # Dimension 2
    size2 = x_shape[2]
    coord2 = temp % size2
    temp //= size2
    if dim != 2:
        offset += coord2 * x_strides[2]

    # Dimension 1
    size1 = x_shape[1]
    coord1 = temp % size1
    temp //= size1
    if dim != 1:
        offset += coord1 * x_strides[1]

    # Dimension 0
    size0 = x_shape[0]
    coord0 = temp % size0
    if dim != 0:
        offset += coord0 * x_strides[0]

    # Sum over the reduction dimension
    sum_val = 0.0
    stride_dim = x_strides[dim]
    
    # Loop over blocks of the reduction dimension
    for block_start in range(0, dim_size, BLOCK_SIZE_REDUCE):
        block_size = min(BLOCK_SIZE_REDUCE, dim_size - block_start)
        offsets = offset + (block_start + tl.arange(0, BLOCK_SIZE_REDUCE)) * stride_dim
        mask = (block_start + tl.arange(0, BLOCK_SIZE_REDUCE)) < dim_size
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)

    # Store the mean
    tl.store(out_ptr + pid, sum_val / dim_size)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    out_shape = list(x.shape)
    out_shape.pop(dim)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    x_strides = list(x.stride())
    x_shape = list(x.shape)
    dim_size = x.shape[dim]
    num_elements_out = out.numel()
    
    BLOCK_SIZE_REDUCE = 128
    grid = (num_elements_out,)
    
    mean_kernel[grid](x, out, x_strides, x_shape, dim, dim_size, num_elements_out, BLOCK_SIZE_REDUCE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)