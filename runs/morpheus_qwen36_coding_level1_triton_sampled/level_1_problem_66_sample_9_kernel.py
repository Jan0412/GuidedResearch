import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, C_out, D_in, H_in, W_in,
    D_k, H_k, W_k,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    D_out, H_out, W_out,
    BLOCK_N: tl.constexpr, BLOCK_C_OUT: tl.constexpr, BLOCK_D_OUT: tl.constexpr, BLOCK_H_OUT: tl.constexpr, BLOCK_W_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr, BLOCK_D_K: tl.constexpr, BLOCK_H_K: tl.constexpr, BLOCK_W_K: tl.constexpr
):
    # Grid mapping: each program handles a tile of output
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d_out = tl.program_id(2)
    pid_h_out = tl.program_id(3)
    pid_w_out = tl.program_id(4)

    # Check bounds
    if pid_n >= N or pid_c_out >= C_out or pid_d_out >= D_out // BLOCK_D_OUT or pid_h_out >= H_out // BLOCK_H_OUT or pid_w_out >= W_out // BLOCK_W_OUT:
        return

    # Output coordinates
    d_out = pid_d_out * BLOCK_D_OUT + tl.arange(0, BLOCK_D_OUT)
    h_out = pid_h_out * BLOCK_H_OUT + tl.arange(0, BLOCK_H_OUT)
    w_out = pid_w_out * BLOCK_W_OUT + tl.arange(0, BLOCK_W_OUT)

    # Mask for output bounds
    mask_d_out = d_out < D_out
    mask_h_out = h_out < H_out
    mask_w_out = w_out < W_out
    mask_out = mask_d_out[:, None, None] & mask_h_out[None, :, None] & mask_w_out[None, None, :]

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_D_OUT, BLOCK_H_OUT, BLOCK_W_OUT), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for c_in in range(0, C_in, BLOCK_C_IN):
        for d_k in range(0, D_k, BLOCK_D_K):
            for h_k in range(0, H_k, BLOCK_H_K):
                for w_k in range(0, W_k, BLOCK_W_K):
                    # Load weight tile
                    w_off = (pid_c_out * C_in + c_in + tl.arange(0, BLOCK_C_IN)[:, None, None, None, None]) * (D_k * H_k * W_k) + \
                             (d_k + tl.arange(0, BLOCK_D_K)[:, None, None, None]) * (H_k * W_k) + \
                             (h_k + tl.arange(0, BLOCK_H_K)[:, None, None]) * W_k + \
                             (w_k + tl.arange(0, BLOCK_W_K)[:, None])
                    mask_w = (c_in + tl.arange(0, BLOCK_C_IN)[:, None, None, None, None]) < C_in & \
                             (d_k + tl.arange(0, BLOCK_D_K)[:, None, None, None]) < D_k & \
                             (h_k + tl.arange(0, BLOCK_H_K)[:, None, None]) < H_k & \
                             (w_k + tl.arange(0, BLOCK_W_K)[:, None]) < W_k
                    w = tl.load(weight_ptr + w_off, mask=mask_w, other=0.0)

                    # Load input tile
                    d_in = d_out * stride_d - padding_d + d_k + tl.arange(0, BLOCK_D_K)[:, None, None, None]
                    h_in = h_out * stride_h - padding_h + h_k + tl.arange(0, BLOCK_H_K)[:, None, None]
                    w_in = w_out * stride_w - padding_w + w_k + tl.arange(0, BLOCK_W_K)[:, None]
                    i_off = (pid_n * C_in + c_in + tl.arange(0, BLOCK_C_IN)[:, None, None, None, None]) * (D_in * H_in * W_in) + \
                            (d_in[:, None, None, None, None]) * (H_in * W_in) + \
                            (h_in[None, :, None, None, None]) * W_in + \
                            (w_in[None, None, :, None, None])
                    mask_i = (c_in + tl.arange(0, BLOCK_C_IN)[:, None, None, None, None]) < C_in & \
                             (d_in[:, None, None, None, None]) >= 0 & (d_in[:, None, None, None, None]) < D_in & \
                             (h_in[None, :, None, None, None]) >= 0 & (h_in[None, :, None, None, None]) < H_in & \
                             (w_in[None, None, :, None, None]) >= 0 & (w_in[None, None, :, None, None]) < W_in
                    i = tl.load(input_ptr + i_off, mask=mask_i, other=0.0)

                    # Accumulate
                    acc += tl.sum(w * i, axis=(0, 1, 2, 3))

    # Apply bias and store
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + pid_c_out)
    out_off = (pid_n * C_out + pid_c_out) * (D_out * H_out * W_out) + d_out[:, None, None] * (H_out * W_out) + h_out[None, :, None] * W_out + w_out[None, None, :]
    tl.store(output_ptr + out_off, acc, mask=mask_out)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, D_k, H_k, W_k = weight.shape
    D_out = (D_in + 2 * padding[0] - dilation[0] * (D_k - 1) - 1) // stride[0] + 1
    H_out = (H_in + 2 * padding[1] - dilation[1] * (H_k - 1) - 1) // stride[1] + 1
    W_out = (W_in + 2 * padding[2] - dilation[2] * (W_k - 1) - 1) // stride[2] + 1

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=torch.float32)

    BLOCK_N = 1
    BLOCK_C_OUT = 1
    BLOCK_D_OUT = 4
    BLOCK_H_OUT = 4
    BLOCK_W_OUT = 4
    BLOCK_C_IN = 8
    BLOCK_D_K = 2
    BLOCK_H_K = 2
    BLOCK_W_K = 2

    grid = (N, C_out, (D_out + BLOCK_D_OUT - 1) // BLOCK_D_OUT, (H_out + BLOCK_H_OUT - 1) // BLOCK_H_OUT, (W_out + BLOCK_W_OUT - 1) // BLOCK_W_OUT)
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, D_in, H_in, W_in,
        D_k, H_k, W_k,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        D_out, H_out, W_out,
        BLOCK_N, BLOCK_C_OUT, BLOCK_D_OUT, BLOCK_H_OUT, BLOCK_W_OUT,
        BLOCK_C_IN, BLOCK_D_K, BLOCK_H_K, BLOCK_W_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        return triton_conv3d(x, weight, bias, self.conv3d.stride, self.conv3d.padding, self.conv3d.dilation, self.conv3d.groups)