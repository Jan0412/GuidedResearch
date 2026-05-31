import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    stride_in,
    stride_out,
    B,
    C,
    L,
    L_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    base_x = x_ptr + pid_b * stride_in + pid_c * L
    base_out = out_ptr + pid_b * stride_out + pid_c * L_out

    l_out_offsets = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_l = l_out_offsets < L_out

    res = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)

    for k in range(kernel_size):
        d = k * dilation
        idx = l_out_offsets * stride - padding + d
        mask_valid = (idx >= 0) & (idx < L)
        val = tl.load(base_x + idx, mask=mask_valid, other=float("-inf"))
        res = tl.maximum(res, val)

    tl.store(base_out + l_out_offsets, res, mask=mask_l)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    B, C, L = x.shape
    L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    num_blocks = (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    stride_in = C * L
    stride_out = C * L_out

    grid = lambda meta: (B, C, num_blocks)
    maxpool1d_kernel[grid](
        x, out, stride_in, stride_out, B, C, L, L_out,
        kernel_size, stride, padding, dilation, BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            raise NotImplementedError("return_indices=True is not supported in this optimized kernel.")
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)