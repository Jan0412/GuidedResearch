import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr, out_ptr,
    stride_x0, stride_x1, stride_x2,
    B, D, K,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pk = B * K
    if pid >= num_pk:
        return
    
    b_idx = pid // K
    k_idx = pid % K
    
    x_base = b_idx * stride_x0 + k_idx * stride_x2
    
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE_D):
        offsets = d + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        x_vals = tl.load(x_ptr + x_base + offsets * stride_x1, mask=mask, other=0.0)
        acc += tl.sum(x_vals)
        
    tl.store(out_ptr + b_idx * K + k_idx, acc)


def triton_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    B, D, K = x.shape
    assert dim == 1, "Kernel optimized for dim=1"
    
    out = torch.empty((B, 1, K), dtype=x.dtype, device=x.device)
    
    stride_x0 = D * K
    stride_x1 = K
    stride_x2 = 1
    
    BLOCK_SIZE_D = 128
    
    grid = (B * K,)
    sum_reduce_kernel[grid](
        x, out,
        stride_x0, stride_x1, stride_x2,
        B, D, K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=4
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)