import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
    B, C_in, C_out, H, W, KH, KW, H_out, W_out,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid mapping: (b, h, w, c_block)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c_block = tl.program_id(3)

    c_start = c_block * BLOCK_C
    c_end = tl.minimum(c_start + BLOCK_C, C_out)
    c = c_start + tl.arange(0, BLOCK_C)
    c_mask = c < C_out

    # Precompute strides for x and out
    stride_x_c = H * W
    stride_x_b = C_in * H * W
    stride_out_c = H_out * W_out
    stride_out_b = C_out * H_out * W_out

    # Precompute strides for w
    stride_w_c = (C_in // groups) * KH * KW
    stride_w_k = KH * KW

    # Group sizes
    group_size_in = C_in // groups
    group_size_out = C_out // groups

    # Base input coordinates
    h_in_base = h * stride_h - pad_h
    w_in_base = w * stride_w - pad_w

    # Accumulator
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Total reduction size
    K = group_size_in * KH * KW

    # Loop over reduction dimension
    for k in range(0, K, BLOCK_K):
        k_idx = tl.arange(0, BLOCK_K)
        k = k + k_idx
        k_mask = k < K

        # Decompose k into kh, kw, c_in_rel
        c_in_rel = k % group_size_in
        k_temp = k // group_size_in
        kh = k_temp % KH
        kw = k_temp // KH

        # Compute input channel index based on group
        g = c // group_size_out
        c_in = g * group_size_in + c_in_rel

        # Compute input coordinates
        h_in = h_in_base + kh * dil_h
        w_in = w_in_base + kw * dil_w

        # Mask for valid input coordinates
        in_mask = k_mask & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

        # Compute pointers
        # Weight pointer: w[c, c_in_rel, kh, kw]
        # c is vector, c_in_rel is vector, kh, kw are scalars
        w_ptr_offset = c * stride_w_c + c_in_rel * stride_w_k + kh * KW + kw
        weight_ptr = w_ptr + w_ptr_offset

        # Input pointer: x[b, c_in, h_in, w_in]
        # b is scalar, c_in is vector, h_in, w_in are vectors
        x_ptr_offset = b * stride_x_b + c_in * stride_x_c + h_in * W + w_in
        input_ptr = x_ptr + x_ptr_offset

        # Load data
        w_val = tl.load(weight_ptr, mask=in_mask, other=0.0)
        x_val = tl.load(input_ptr, mask=in_mask, other=0.0)

        # Accumulate
        acc += w_val * x_val

    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + c, mask=c_mask, other=0.0)
        acc += bias

    # Store result
    out_ptr_offset = b * stride_out_b + c * stride_out_c + h * W_out + w
    tl.store(out_ptr + out_ptr_offset, acc, mask=c_mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None,
                  stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    B, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Compute output dimensions
    H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    # Tunable block sizes
    BLOCK_C = 32
    BLOCK_K = 64

    # Grid calculation
    grid = (B, H_out, W_out, (C_out + BLOCK_C - 1) // BLOCK_C)

    conv2d_kernel[grid](
        x, w, b, out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        B, C_in, C_out, H, W, KH, KW, H_out, W_out,
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)