import torch
import torch.nn as nn
import math
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    out_D, out_H, out_W,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    od = tl.program_id(2)
    oh = tl.program_id(3)
    ow = tl.program_id(4)
    
    stride_D = H * W
    stride_H = W
    stride_C = D * H * W
    stride_N = C * D * H * W
    
    ptr_nc = x_ptr + n * stride_N + c * stride_C
    
    out_stride_D = out_H * out_W
    out_stride_H = out_W
    
    out_offset = ((n * C + c) * out_D + od) * out_stride_D + oh * out_stride_H + ow
    out_ptr = y_ptr + out_offset
    
    start_d = od * stride - padding
    start_h = oh * stride - padding
    start_w = ow * stride - padding
    
    max_val = float('-inf')
    
    for kd in range(kernel_size):
        id = start_d + kd * dilation
        for kh in range(kernel_size):
            ih = start_h + kh * dilation
            for kw in range(kernel_size):
                iw = start_w + kw * dilation
                
                if 0 <= id < D and 0 <= ih < H and 0 <= iw < W:
                    ptr = ptr_nc + id * stride_D + ih * stride_H + iw
                    val = tl.load(ptr)
                    if val > max_val:
                        max_val = val
    
    tl.store(out_ptr, max_val)


def triton_maxpool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, ceil_mode: bool = False):
    assert x.is_cuda
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    
    def out_size(dim):
        if ceil_mode:
            return math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        else:
            return (dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            
    out_D = out_size(D)
    out_H = out_size(H)
    out_W = out_size(W)
    
    y = torch.empty(N, C, out_D, out_H, out_W, dtype=x.dtype, device=x.device)
    
    grid = (N, C, out_D, out_H, out_W)
    
    maxpool3d_kernel[grid](
        x, y,
        N, C, D, H, W,
        out_D, out_H, out_W,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=128
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        # Note: return_indices is not supported by the custom Triton kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)