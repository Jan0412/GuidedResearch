import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, D, H, W,
    C_out, C_in_g, KD, KH, KW,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    groups,
    has_bias,
):
    # pid_0: b * C_out + co
    # pid_1: d_out * (H_out * W_out) + h_out * W_out + w_out
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    b = pid_0 // C_out
    co = pid_0 % C_out

    d_out = pid_1 // (H_out * W_out)
    rem = pid_1 % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    # Calculate input channel start for the current group
    c_out_g = C_out // groups
    group_idx = co // c_out_g
    c_in_start = group_idx * C_in_g

    acc = 0.0

    # Direct convolution loops
    # Triton compiles these Python loops into efficient CUDA loops
    for ci in range(C_in_g):
        curr_ci = c_in_start + ci
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    d_in = d_out * stride_d + kd * dil_d - pad_d
                    h_in = h_out * stride_h + kh * dil_h - pad_h
                    w_in = w_out * stride_w + kw * dil_w - pad_w

                    if d_in >= 0 and d_in < D and h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        # Index for input x: [b, curr_ci, d_in, h_in, w_in]
                        x_off = b * (C_in * D * H * W) + \
                                curr_ci * (D * H * W) + \
                                d_in * (H * W) + \
                                h_in * W + \
                                w_in
                        
                        # Index for weight w: [co, ci, kd, kh, kw]
                        w_off = co * (C_in_g * KD * KH * KW) + \
                                ci * (KD * KH * KW) + \
                                kd * (KH * KW) + \
                                kh * KW + \
                                kw
                        
                        acc += tl.load(x_ptr + x_off) * tl.load(w_ptr + w_off)

    if has_bias:
        acc += tl.load(b_ptr + co)

    # Index for output: [b, co, d_out, h_out, w_out]
    out_off = b * (C_out * D_out * H_out * W_out) + \
              co * (D_out * H_out * W_out) + \
              d_out * (H_out * W_out) + \
              h_out * W_out + \
              w_out
    
    tl.store(out_ptr + out_off, acc)


def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # x: (B, C_in, D, H, W)
    # weight: (C_out, C_in_g, KD, KH, KW)
    B, C_in, D, H, W = x.shape
    C_out, C_in_g, KD, KH, KW = weight.shape
    
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    D_out = (D + 2 * pd - dd * (KD - 1) - 1) // sd + 1
    H_out = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    out = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    grid = (B * C_out, D_out * H_out * W_out)

    conv3d_kernel[grid](
        x, weight, bias if bias is not None else 0, out,
        B, C_in, D, H, W,
        C_out, C_in_g, KD, KH, KW,
        D_out, H_out, W_out,
        sd, sh, sw,
        pd, ph, pw,
        dd, dh, dw,
        groups,
        bias is not None
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv3d to manage the parameters (weights and bias)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton implementation instead of the built-in Conv3d
        return triton_conv3d(
            x, 
            self.conv3d.weight, 
            self.conv3d.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )