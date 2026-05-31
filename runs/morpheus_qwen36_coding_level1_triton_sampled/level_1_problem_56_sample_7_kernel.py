import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    in_channels,
    out_channels,
    kH,
    kW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    batch,
    height,
    width,
    height_out,
    width_out,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_c = tl.program_id(4)

    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W
    start_c = pid_c * BLOCK_C

    offsets_h = start_h + tl.arange(0, BLOCK_H)
    offsets_w = start_w + tl.arange(0, BLOCK_W)
    offsets_c = start_c + tl.arange(0, BLOCK_C)

    mask_h = offsets_h < height_out
    mask_w = offsets_w < width_out
    mask_c = offsets_c < (in_channels // groups)

    # Global mask for output elements
    mask_out = mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]

    # Base indices for input and output
    base_h = offsets_h * stride_h - pad_h
    base_w = offsets_w * stride_w - pad_w

    # Channel range for this output channel considering groups
    c_start_global = pid_out * (in_channels // groups)

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Loop over kernel height
    for kh in range(kH):
        # Loop over kernel width
        for kw in range(kW):
            # Compute input coordinates
            ih = base_h + kh * dilation_h
            iw = base_w + kw * dilation_w

            # Input mask for spatial bounds
            mask_in_h = (ih >= 0) & (ih < height)
            mask_in_w = (iw >= 0) & (iw < width)
            mask_in_spatial = mask_in_h[:, None, None] & mask_in_w[None, :, None]
            mask_in = mask_in_spatial & mask_c[None, None, :]

            # Load input tensor
            # x_ptr offset: batch * in_channels * height * width + c * height * width + ih * width + iw
            x_offsets = (
                pid_batch * in_channels * height * width +
                (c_start_global + offsets_c) * height * width +
                ih[:, None, None] * width +
                iw[None, :, None]
            )
            x = tl.load(x_ptr + x_offsets, mask=mask_in, other=0.0)

            # Load weight tensor
            # w_ptr offset: out_channels * (in_channels // groups) * kH * kW + kh * (in_channels // groups) * kW + kw * (in_channels // groups) + c
            # Note: w_ptr is contiguous in memory as (out_channels, in_channels // groups, kH, kW)
            w_offsets = (
                pid_out * (in_channels // groups) * kH * kW +
                kh * (in_channels // groups) * kW +
                kw * (in_channels // groups) +
                offsets_c[None, None, :]
            )
            w = tl.load(w_ptr + w_offsets, mask=mask_out, other=0.0)

            acc += x * w

    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_out)
        acc += bias_val

    # Store result
    # out_ptr offset: batch * out_channels * height_out * width_out + pid_out * height_out * width_out + h * width_out + w
    out_offsets = (
        pid_batch * out_channels * height_out * width_out +
        pid_out * height_out * width_out +
        offsets_h[:, None, None] * width_out +
        offsets_w[None, :, None]
    )
    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


def triton_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kH, kW = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    height_out = (height + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    width_out = (width + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=x.dtype, device=x.device)

    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 8

    grid = (
        batch_size,
        out_channels,
        triton.cdiv(height_out, BLOCK_H),
        triton.cdiv(width_out, BLOCK_W),
        triton.cdiv(in_channels // groups, BLOCK_C),
    )

    conv2d_kernel[grid](
        x, w, b, out,
        in_channels, out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
        batch_size, height, width,
        height_out, width_out,
        BLOCK_H, BLOCK_W, BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )