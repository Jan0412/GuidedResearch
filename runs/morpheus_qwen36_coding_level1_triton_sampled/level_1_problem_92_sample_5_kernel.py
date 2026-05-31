import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    out_ptr,
    shape_ptr,
    stride_ptr,
    dim,
    n_elements,
):
    pid = tl.program_id(0)
    
    # Compute base offset for the current slice
    offset = 0
    temp_pid = pid
    ndim = tl.load(shape_ptr)
    
    # Decompose pid into indices for non-dim dimensions
    for k in range(ndim - 1, -1, -1):
        if k == dim:
            continue
        shape_k = tl.load(shape_ptr + k)
        stride_k = tl.load(stride_ptr + k)
        idx = temp_pid % shape_k
        temp_pid = temp_pid // shape_k
        offset += idx * stride_k
        
    stride_dim = tl.load(stride_ptr + dim)
    shape_dim = tl.load(shape_ptr + dim)
    
    # Compute exclusive cumsum along dim
    running_sum = 0.0
    for i in range(shape_dim):
        offset_i = offset + i * stride_dim
        val = tl.load(x_ptr + offset_i)
        tl.store(out_ptr + offset_i, running_sum)
        running_sum += val


def triton_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    # Prepare shape and stride arrays on device
    shape_tensor = torch.tensor(list(x.shape), dtype=torch.int32, device=x.device)
    stride_tensor = torch.tensor(list(x.stride()), dtype=torch.int32, device=x.device)
    
    n_elements = x.numel()
    shape_dim = x.shape[dim]
    num_slices = n_elements // shape_dim
    
    grid = (num_slices,)
    
    exclusive_cumsum_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        shape_ptr=shape_tensor.data_ptr(),
        stride_ptr=stride_tensor.data_ptr(),
        dim=dim,
        n_elements=n_elements,
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum using a custom Triton kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)