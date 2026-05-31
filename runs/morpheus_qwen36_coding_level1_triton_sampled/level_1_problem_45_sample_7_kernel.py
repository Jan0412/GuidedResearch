import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr, y_ptr,
    B, C, H, W, K, S, P,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_h = (H + 2 * P - K) // S + 1
    out_w = (W + 2 * P - K) // S + 1
    num_out = B * C * out_h * out_w
    
    if pid >= num_out:
        return
        
    bc = pid // (out_h * out_w)
    b = bc // C
    c = bc % C
    oh_ow = pid % (out_h * out_w)
    oh = oh_ow // out_w
    ow = oh_ow % out_w
    
    offsets = tl.arange(0, BLOCK_SIZE)
    k = offsets // K
    l = offsets % K
    row_pad = oh * S + k + P
    col_pad = ow * S + l + P
    row_orig = row_pad - P
    col_orig = col_pad - P
    
    mask = (offsets < K * K) & (row_orig >= 0) & (row_orig < H) & (col_orig >= 0) & (col_orig < W)
    
    linear_offset = row_orig * W + col_orig
    base_ptr = x_ptr + b * C * H * W + c * H * W
    ptr = base_ptr + linear_offset
    
    window = tl.load(ptr, mask=mask, other=0.0)
    acc = tl.sum(window)
    acc /= (K * K)
    
    out_ptr = y_ptr + b * C * out_h * out_w + c * out_h * out_w + oh * out_w + ow
    tl.store(out_ptr, acc)


def triton_avg_pool2d(x, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    x = x.contiguous()
    B, C, H, W = x.shape
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    y = torch.empty((B, C, out_h, out_w), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (B * C * out_h * out_w,)
    
    avg_pool2d_kernel[grid](x, y, B, C, H, W, kernel_size, stride, padding, BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)