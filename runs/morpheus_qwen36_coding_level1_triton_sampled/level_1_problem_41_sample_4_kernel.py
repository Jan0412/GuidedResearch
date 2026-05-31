import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr, y_ptr,
    B, C, L, OutL,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    feat_idx = tl.program_id(1)
    block_idx = tl.program_id(2)
    
    x_base = x_ptr + batch_idx * C * L + feat_idx * L
    y_base = y_ptr + batch_idx * C * OutL + feat_idx * OutL
    
    out_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < OutL
    
    in_start = padding + block_idx * BLOCK_SIZE * stride
    
    max_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    for i in range(kernel_size):
        in_offsets = in_start + out_offsets * stride + i * dilation
        valid_mask = (in_offsets >= 0) & (in_offsets < L)
        vals = tl.load(x_base + in_offsets, mask=valid_mask, other=0.0)
        max_vals = tl.maximum(max_vals, vals)
        
    tl.store(y_base + out_offsets, max_vals, mask=mask)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    B, C, L = x.shape
    OutL = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    y = torch.empty((B, C, OutL), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (B, C, (OutL + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    
    maxpool1d_kernel[grid](x, y, B, C, L, OutL, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)