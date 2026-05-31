import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    in_channels, out_channels, kernel_size, stride, padding, output_padding,
    height, width, height_out, width_out,
    groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Grid: (N, C_out, H_out_tiles, W_out_tiles)
    n_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    h_tile_idx = tl.program_id(2)
    w_tile_idx = tl.program_id(3)

    # Calculate block offsets
    h_start = h_tile_idx * BLOCK_SIZE_H
    w_start = w_tile_idx * BLOCK_SIZE_W

    # Create offsets for the tile
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    # Mask for valid output coordinates
    h_mask = h_offsets < height_out
    w_mask = w_offsets < width_out
    out_mask = h_mask[:, None] & w_mask[None, :]

    # Determine group for this output channel
    out_per_group = out_channels // groups
    group_idx = c_out_idx // out_per_group
    in_per_group = in_channels // groups

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over kernel dimensions
    for kh in tl.range(0, kernel_size, BLOCK_SIZE_K):
        kh_offsets = kh + tl.arange(0, BLOCK_SIZE_K)
        kh_mask = kh_offsets < kernel_size

        for kw in tl.range(0, kernel_size, BLOCK_SIZE_K):
            kw_offsets = kw + tl.arange(0, BLOCK_SIZE_K)
            kw_mask = kw_offsets < kernel_size

            # Compute input coordinates
            # ih = (oh + padding - kh) / stride
            # We need integer division. Triton handles this.
            ih = (h_offsets + padding - kh_offsets[None, :]) // stride
            iw = (w_offsets + padding - kw_offsets[None, :]) // stride

            # Mask for valid input coordinates
            ih_mask = (ih >= 0) & (ih < height)
            iw_mask = (iw >= 0) & (iw < width)
            spatial_mask = ih_mask[:, None] & iw_mask[None, :] & kh_mask[None, :] & kw_mask[:, None]

            # Loop over input channels in the group
            for c_in_local in tl.range(0, in_per_group, BLOCK_SIZE_C):
                c_in_offsets = c_in_local + tl.arange(0, BLOCK_SIZE_C)
                c_in_mask = c_in_offsets < in_per_group

                # Global input channel index
                c_in_global = group_idx * in_per_group + c_in_offsets[None, None]
                c_in_mask_global = c_in_mask[None, None]

                # Load input
                x_ptr_offset = n_idx * in_channels * height * width + c_in_global * height * width + ih[:, None, None] * width + iw[None, :, None]
                x = tl.load(x_ptr + x_ptr_offset, mask=spatial_mask[:, :, None] & c_in_mask_global, other=0.0)

                # Load weight
                # weight shape: (C_out, C_in//groups, kH, kW)
                weight_ptr_offset = c_out_idx * in_per_group * kernel_size * kernel_size + c_in_global * kernel_size * kernel_size + kh_offsets[None, None, :] * kernel_size + kw_offsets[None, None, :]
                w = tl.load(weight_ptr + weight_ptr_offset, mask=spatial_mask[:, :, None] & c_in_mask_global, other=0.0)

                # Accumulate
                acc += x * w

    # Add bias if available
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_out_idx)
        acc += bias

    # Store output
    out_ptr_offset = n_idx * out_channels * height_out * width_out + c_out_idx * height_out * width_out + h_offsets[:, None] * width_out + w_offsets[None, :]
    tl.store(out_ptr + out_ptr_offset, acc, mask=out_mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    n, c_in, h, w = x.shape
    c_out, c_in_w, k_h, k_w = weight.shape
    assert c_in == c_in_w * groups, "Input channels must match weight channels times groups"

    # Compute output shape
    h_out = (h - 1) * stride - 2 * padding + k_h + output_padding + 1
    w_out = (w - 1) * stride - 2 * padding + k_w + output_padding + 1

    out = torch.empty((n, c_out, h_out, w_out), dtype=x.dtype, device=x.device)

    # Block sizes (tunable)
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_K = 1  # Kernel size is small, loop is fine
    BLOCK_SIZE_C = 16

    # Grid
    grid = (n, c_out, (h_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (w_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        c_in, c_out, k_h, stride, padding, output_padding,
        h, w, h_out, w_out,
        groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.conv_transpose2d.stride[0],
            padding=self.conv_transpose2d.padding[0],
            output_padding=self.conv_transpose2d.output_padding[0],
            groups=self.conv_transpose2d.groups
        )