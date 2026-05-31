import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in, D, H, W,
    C_out, D_out, H_out, W_out,
    kD, kH, kW,
    sD, sH, sW,
    pD, pH, pW,
    oD, oH, oW,
    groups,
    BLOCK_C: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c_block = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    c_start = pid_c_block * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C_out

    # Group and local channel indices
    c_per_group = C_out // groups
    g = c_offsets // c_per_group
    c_local = c_offsets % c_per_group
    
    c_in_per_group = C_in // groups
    c_in_offsets = g * c_in_per_group + c_local
    mask_c_in = c_in_offsets < C_in

    # Base pointers
    x_ptr_b = x_ptr + pid_b * C_in * D * H * W
    w_ptr_base = w_ptr + c_offsets[:, None] * C_in * kD * kH * kW + c_in_offsets[None, :] * kD * kH * kW

    # Accumulator
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over kernel dimensions
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                # Input spatial coordinates
                in_d = pid_d * sD - pD + kd
                in_h = pid_h * sH - pH + kh
                in_w = pid_w * sW - pW + kw

                # Input bounds mask
                mask_in_d = in_d >= 0
                mask_in_h = in_h >= 0
                mask_in_w = in_w >= 0
                mask_in_d = tl.where(in_d < D, mask_in_d, False)
                mask_in_h = tl.where(in_h < H, mask_in_h, False)
                mask_in_w = tl.where(in_w < W, mask_in_w, False)
                mask_in = mask_in_d & mask_in_h & mask_in_w

                # Load input
                x_ptr_in = x_ptr_b + in_d * H * W + in_h * W + in_w
                x_vals = tl.load(
                    x_ptr_in + c_in_offsets * H * W,
                    mask=mask_c_in & mask_in[None, :],
                    other=0.0
                )

                # Load weights
                w_ptr_k = w_ptr_base + kd * kH * kW + kh * kW + kw
                w_vals = tl.load(
                    w_ptr_k,
                    mask=mask_c[:, None] & mask_c_in[None, :],
                    other=0.0
                )

                # Accumulate
                acc += tl.sum(x_vals[None, :] * w_vals, axis=1)

    # Store output
    out_ptr_base = out_ptr + pid_b * C_out * D_out * H_out * W_out + pid_d * H_out * W_out + pid_h * W_out + pid_w
    tl.store(out_ptr_base + c_offsets * H_out * W_out, acc, mask=mask_c)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None,
                            stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0),
                            groups=1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, D, H, W = x.shape
    C_out, C_in_w, kD, kH, kW = w.shape
    assert C_in_w * groups == C_in
    assert C_out % groups == 0

    sD, sH, sW = stride
    pD, pH, pW = padding
    oD, oH, oW = output_padding

    D_out = (D - 1) * sD - 2 * pD + kD + oD
    H_out = (H - 1) * sH - 2 * pH + kH + oH
    W_out = (W - 1) * sW - 2 * pW + kW + oW

    out = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_C = 32
    BLOCK_C_IN = 32

    grid = (B, (C_out + BLOCK_C - 1) // BLOCK_C, D_out, H_out, W_out)

    conv_transpose3d_kernel[grid](
        x, w, out,
        B, C_in, D, H, W,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        oD, oH, oW,
        groups,
        BLOCK_C=BLOCK_C,
        BLOCK_C_IN=BLOCK_C_IN
    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )