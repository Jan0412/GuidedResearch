import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    n_elements,
    in_channels,
    out_channels,
    in_h,
    in_w,
    kernel_h,
    kernel_w,
    stride,
    padding,
    output_padding,
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // (BLOCK_M * BLOCK_N)
    o_h = (pid // BLOCK_N) % BLOCK_M
    o_w = pid % BLOCK_N

    x_ptr += n * in_channels * in_h * in_w
    out_ptr += n * out_channels * BLOCK_M * BLOCK_N

    # Initialize output vector
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over kernel dimensions
    for k_h in tl.static_range(kernel_h):
        for k_w in tl.static_range(kernel_w):
            i_h = o_h * stride + k_h - padding
            i_w = o_w * stride + k_w - padding

            # Check input bounds
            mask_h = i_h >= 0 & i_h < in_h
            mask_w = i_w >= 0 & i_w < in_w
            mask = mask_h & mask_w

            if not tl.sum(mask):
                continue

            # Load input tile
            x_offsets = i_h * in_w * in_channels + i_w * in_channels + tl.arange(0, in_channels)
            x_tile = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

            # Load weight tile
            w_offsets = (k_h * kernel_w * in_channels * out_channels) + (k_w * in_channels * out_channels) + tl.arange(0, out_channels)[:, None] * in_channels + tl.arange(0, in_channels)[None, :]
            w_tile = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)

            # Accumulate
            acc += tl.sum(w_tile * x_tile[None, :], axis=1)

    # Add bias
    if bias_ptr is not None:
        bias_offsets = tl.arange(0, out_channels)
        acc += tl.load(bias_ptr + bias_offsets, mask=tl.arange(0, out_channels) < out_channels)

    # Store output
    out_offsets = o_h * BLOCK_N * out_channels + o_w * out_channels + tl.arange(0, out_channels)
    tl.store(out_ptr + out_offsets, acc, mask=tl.arange(0, out_channels) < out_channels)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_h, in_w = x.shape[2], x.shape[3]
        
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_h + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_w + self.output_padding
        
        out = torch.empty(batch_size, self.out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (batch_size * out_h * out_w,)
        
        conv_transpose2d_kernel[grid](
            x, self.weight, out, self.bias,
            batch_size * out_h * out_w,
            self.in_channels, self.out_channels,
            in_h, in_w,
            self.kernel_h, self.kernel_w,
            self.stride, self.padding, self.output_padding,
            self.groups,
            BLOCK_M=out_h, BLOCK_N=out_w
        )
        return out