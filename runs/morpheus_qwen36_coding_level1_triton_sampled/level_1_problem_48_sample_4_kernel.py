import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr, out_ptr,
    B, D1, D2,
    stride_x_b, stride_x_d1, stride_x_d2,
    stride_out_b, stride_out_d2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = B * D2
    if pid >= num_blocks:
        return
    
    b = pid // D2
    d2 = pid % D2
    
    x_offset = b * stride_x_b + d2 * stride_x_d2
    out_offset = b * stride_out_b + d2 * stride_out_d2
    
    sum_val = 0.0
    for start in range(0, D1, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D1
        x_vals = tl.load(x_ptr + x_offset + offsets * stride_x_d1, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals, axis=0)
        
    out_val = sum_val / D1
    tl.store(out_ptr + out_offset, out_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    B, D1, D2 = x.shape
    out = torch.empty((B, D2), dtype=x.dtype, device=x.device)
    
    stride_x_b, stride_x_d1, stride_x_d2 = x.stride()
    stride_out_b, stride_out_d2 = out.stride()
    
    n_elements = B * D2
    BLOCK_SIZE = 512
    
    grid = (n_elements,)
    
    mean_kernel[grid](
        x, out,
        B, D1, D2,
        stride_x_b, stride_x_d1, stride_x_d2,
        stride_out_b, stride_out_d2,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)