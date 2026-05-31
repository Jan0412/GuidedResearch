import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    IC,
    OC,
    K,
    L,
    L_out,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
):
    b_idx = tl.program_id(0)
    oc_idx = tl.program_id(1)
    out_block_idx = tl.program_id(2)

    out_offsets = out_block_idx * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    mask_out = out_offsets < L_out

    in_offsets = out_offsets[:, None] * stride - padding + tl.arange(0, K)[None, :] * dilation
    mask_in_pos = (in_offsets >= 0) & (in_offsets < L)

    ic_local_offsets = tl.arange(0, BLOCK_SIZE_IN_CH)
    mask_ic_local = ic_local_offsets < (IC // groups)
    ic_global_offsets = ic_local_offsets + (oc_idx // groups) * (IC // groups)

    x_vals = tl.load(
        x_ptr + b_idx * IC * L + ic_global_offsets[:, None] * L + in_offsets,
        mask=mask_ic_local[:, None] & mask_out[None, :] & mask_in_pos,
        other=0.0,
    )

    w_vals = tl.load(
        w_ptr + oc_idx * (IC // groups) * K + ic_local_offsets[:, None] * K + tl.arange(0, K)[None, :],
        mask=mask_ic_local[:, None],
        other=0.0,
    )

    acc = tl.sum(x_vals * w_vals, axis=0)

    if b_ptr is not None:
        acc += tl.load(b_ptr + oc_idx)

    tl.store(
        out_ptr + b_idx * OC * L_out + oc_idx * L_out + out_offsets,
        acc,
        mask=mask_out,
    )


def triton_conv1d(x, w, b, stride, padding, dilation, groups):
    B, IC, L = x.shape
    OC, IC_w, K = w.shape
    assert IC_w * groups == IC, "Input channels must match weight channels * groups"
    L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    out = torch.empty((B, OC, L_out), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_OUT = 64
    BLOCK_SIZE_IN_CH = 16

    grid = (B, OC, (L_out + BLOCK_SIZE_OUT - 1) // BLOCK_SIZE_OUT)
    conv1d_kernel[grid](
        x, w, b, out,
        B, IC, OC, K, L, L_out,
        stride, padding, dilation, groups,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT,
        BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )