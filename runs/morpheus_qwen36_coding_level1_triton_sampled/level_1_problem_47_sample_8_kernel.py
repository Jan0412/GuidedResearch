import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr, y_ptr,
    B, D1, D2,
    stride_x_b, stride_x_d1, stride_x_d2,
    BLOCK_SIZE_D1: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d2 = tl.program_id(1)
    
    base_idx = pid_b * stride_x_b + pid_d2 * stride_x_d2
    
    acc = 0.0
    num_blocks = tl.cdiv(D1, BLOCK_SIZE_D1)
    
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
        mask = offsets < D1
        x_vals = tl.load(x_ptr + base_idx + offsets * stride_x_d1, mask=mask, other=0.0)
        acc += tl.sum(x_vals)
        
    out_idx = pid_b * D2 + pid_d2
    tl.store(y_ptr + out_idx, acc)


def triton_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    B, D1, D2 = x.shape
    assert dim == 1, "Optimized kernel assumes dim=1 reduction"
    
    y = torch.empty((B, 1, D2), dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE_D1 = 256
    
    grid = (B, D2)
    
    sum_reduce_kernel[grid](
        x, y,
        B, D1, D2,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE_D1=BLOCK_SIZE_D1
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)