import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr, out_ptr,
    B, D1, D2,
    stride_b, stride_d1, stride_d2,
    BLOCK_SIZE_D1: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // D2
    d2 = pid % D2
    
    acc = 0.0
    
    for d1 in range(0, D1, BLOCK_SIZE_D1):
        offsets_d1 = d1 + tl.arange(0, BLOCK_SIZE_D1)
        mask = offsets_d1 < D1
        offsets = (b * stride_b + d2 * stride_d2) + offsets_d1 * stride_d1
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x)
        
    tl.store(out_ptr + b * D2 + d2, acc)


def triton_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    if dim != 1:
        raise ValueError("Only dim=1 is supported for this optimized kernel.")
        
    B, D1, D2 = x.shape
    out = torch.empty((B, 1, D2), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_D1 = 1024
    
    grid = (B * D2,)
    
    sum_kernel[grid](x, out, B, D1, D2,
                     x.stride(0), x.stride(1), x.stride(2),
                     BLOCK_SIZE_D1=BLOCK_SIZE_D1)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum(x, self.dim)