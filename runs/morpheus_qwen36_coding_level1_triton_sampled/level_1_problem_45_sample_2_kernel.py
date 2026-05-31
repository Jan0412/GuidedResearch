import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    W: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_oh = tl.cdiv(H, stride) - tl.cdiv(kernel_size - 1, stride)
    num_ow = tl.cdiv(W, stride) - tl.cdiv(kernel_size - 1, stride)
    
    b_c = pid // (num_oh * num_ow)
    h = (pid % (num_oh * num_ow)) // num_ow
    w = pid % num_ow
    
    b = b_c // C
    c = b_c % C
    
    base_offset = (b * C + c) * H * W
    
    start_r = h * stride
    start_c = w * stride
    
    row_offsets = tl.arange(0, BLOCK_SIZE_H)[:, None] * W
    col_offsets = tl.arange(0, BLOCK_SIZE_W)[None, :]
    
    offsets = base_offset + start_r * W + start_c + row_offsets + col_offsets
    
    mask = (row_offsets < kernel_size) & (col_offsets < kernel_size)
    
    x_win = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    out_val = tl.sum(x_win, axis=(0, 1)) / (kernel_size * kernel_size)
    
    out_offset = pid
    tl.store(out_ptr + out_offset, out_val)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be contiguous FP32 CUDA tensor."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    
    if stride is None:
        stride = kernel_size
    
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((B, C, out_h, out_w), dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE_H = kernel_size
    BLOCK_SIZE_W = kernel_size
    
    grid = (B * C * out_h * out_w,)
    
    avg_pool_kernel[grid](
        x, out, W, H, C, B, kernel_size, stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)