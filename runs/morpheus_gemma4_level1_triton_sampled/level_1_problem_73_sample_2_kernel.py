import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, D, H, W,
    Cout, kD, kH, kW,
    stride, padding, groups,
    Dout, Hout, Wout,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    n = tl.program_id(0)
    co = tl.program_id(1)
    do = tl.program_id(2)
    ho = tl.program_id(3)
    wo_start = tl.program_id(4) * BLOCK_W

    # Output channel group calculations
    cout_per_group = Cout // groups
    group = co // cout_per_group
    co_in_group = co % cout_per_group
    ci_start = group * (Cin // groups)
    ci_end = (group + 1) * (Cin // groups)

    # Width offsets for vectorization
    wo_offsets = wo_start + tl.arange(0, BLOCK_W)
    mask_wo = wo_offsets < Wout

    # Accumulator for the output elements
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels in the group
    for ci in range(ci_start, ci_end):
        # Loop over kernel dimensions
        for kd in range(kD):
            di = (do + padding - kd) // stride
            # Check if di is valid and fits the stride condition
            if (do + padding - kd) % stride == 0 and 0 <= di < D:
                for kh in range(kH):
                    hi = (ho + padding - kh) // stride
                    # Check if hi is valid and fits the stride condition
                    if (ho + padding - kh) % stride == 0 and 0 <= hi < H:
                        for kw in range(kW):
                            # Calculate input width index
                            wi = (wo_offsets + padding - kw) // stride
                            
                            # Mask for: 
                            # 1. wo_offsets within output bounds
                            # 2.Stride condition for width
                            # 3. wi within input bounds
                            mask = mask_wo & ((wo_offsets + padding - kw) % stride == 0) & (wi >= 0) & (wi < W)
                            
                            # Calculate pointers
                            # x: (B, Cin, D, H, W)
                            x_idx = n * (Cin * D * H * W) + ci * (D * H * W) + di * (H * W) + hi * W + wi
                            # w: (Cin, Cout_per_group, kD, kH, kW)
                            w_idx = ci * (cout_per_group * kD * kH * kW) + co_in_group * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw
                            
                            val_x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                            val_w = tl.load(w_ptr + w_idx)
                            acc += val_x * val_w

    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + co)
        acc += bias

    # Store the result
    out_idx = n * (Cout * Dout * Hout * Wout) + co * (Dout * Hout * Wout) + do * (Hout * Wout) + ho * Wout + wo_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_wo)


def triton_conv_transpose3d(x, weight, bias, stride, padding, groups, output_padding=0):
    # Input shapes
    B, Cin, D, H, W = x.shape
    Cin_w, Cout_per_group, kD, kH, kW = weight.shape
    Cout = Cout_per_group * groups

    # Calculate output dimensions
    Dout = (D - 1) * stride - 2 * padding + kD + output_padding
    Hout = (H - 1) * stride - 2 * padding + kH + output_padding
    Wout = (W - 1) * stride - 2 * padding + kW + output_padding

    x = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    if bias is not None:
        bias = bias.contiguous().cuda()

    out = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)

    BLOCK_W = 16
    grid = (B, Cout, Dout, Hout, (Wout + BLOCK_W - 1) // BLOCK_W)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, D, H, W,
        Cout, kD, kH, kW,
        stride, padding, groups,
        Dout, Hout, Wout,
        BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the ConvTranspose3d layer to manage parameters (weight and bias)
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton implementation instead of the PyTorch operator
        return triton_conv_transpose3d(
            x, 
            self.conv_transpose3d.weight, 
            self.conv_transpose3d.bias, 
            self.stride, 
            self.padding, 
            self.groups, 
            self.output_padding
        )