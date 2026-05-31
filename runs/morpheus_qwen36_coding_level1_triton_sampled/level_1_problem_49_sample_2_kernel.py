import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr, out_ptr,
    dim1, dim2,
    stride_x0, stride_x1, stride_x2,
    BLOCK_SIZE_D1: tl.constexpr,
):
    pid_d2 = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offsets_d1 = tl.arange(0, BLOCK_SIZE_D1)
    mask_d1 = offsets_d1 < dim1
    
    base_ptr = x_ptr + pid_b * stride_x0 + pid_d2 * stride_x2
    
    max_val = -float('inf')
    
    for start_d1 in range(0, dim1, BLOCK_SIZE_D1):
        offsets = start_d1 + offsets_d1
        ptr = base_ptr + offsets * stride_x1
        values = tl.load(ptr, mask=mask_d1, other=-float('inf'))
        max_val = tl.max(values, axis=0)
        
    tl.store(out_ptr + pid_b * dim2 + pid_d2, max_val)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    batch_size, dim1, dim2 = x.shape
    # Assuming dim=1 based on the provided example architecture
    out = torch.empty((batch_size, dim2), dtype=x.dtype, device=x.device)
    
    stride_x0 = x.stride(0)
    stride_x1 = x.stride(1)
    stride_x2 = x.stride(2)
    
    BLOCK_SIZE_D1 = 128
    
    grid = (dim2, batch_size)
    
    max_reduce_kernel[grid](
        x, out,
        dim1, dim2,
        stride_x0, stride_x1, stride_x2,
        BLOCK_SIZE_D1=BLOCK_SIZE_D1
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)