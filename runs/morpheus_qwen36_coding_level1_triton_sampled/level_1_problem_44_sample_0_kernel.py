import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr, y_ptr,
    B, C, L, L_out,
    kernel_size: tl.constexpr,
    stride, padding
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    
    start = pid2 * stride - padding
    offsets = start + tl.arange(0, kernel_size)
    mask = (offsets >= 0) & (offsets < L)
    
    x_vals = tl.load(x_ptr + pid0 * C * L + pid1 * L + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x_vals)
    
    tl.store(y_ptr + pid0 * C * L_out + pid1 * L_out + pid2, sum_val / kernel_size)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    x = x.contiguous()
    B, C, L = x.shape
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    y = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)
    
    grid = (B, C, L_out)
    
    avg_pool1d_kernel[grid](
        x, y,
        B, C, L, L_out,
        kernel_size, stride, padding
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)