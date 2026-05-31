import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_n, stride_c, stride_h, stride_w,
    kernel_size,
    stride, padding, output_padding,
    groups,
    in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Output element coordinates
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_n = tl.program_id(3)

    # Block coordinates
    block_h = pid_h * BLOCK_SIZE_H
    block_w = pid_w * BLOCK_SIZE_W
    block_c = pid_c * BLOCK_SIZE_C
    block_n = pid_n

    # Output coordinates for this block
    h_out = block_h + tl.arange(0, BLOCK_SIZE_H)
    w_out = block_w + tl.arange(0, BLOCK_SIZE_W)
    c_out = block_c + tl.arange(0, BLOCK_SIZE_C)

    # Mask for valid output elements
    mask_h = h_out < height_out
    mask_w = w_out < width_out
    mask_c = c_out < out_channels
    mask_out = mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]

    # Group index
    group_idx = pid_c // (out_channels // groups)
    c_in_group = c_out % (out_channels // groups)

    # Input channel range for this group
    c_in_start = group_idx * (in_channels // groups)
    c_in_end = c_in_start + (in_channels // groups)
    c_in = tl.arange(0, BLOCK_SIZE_C) + c_in_start

    # Mask for valid input channels
    mask_c_in = c_in < in_channels

    # Load weights for this group and channel block
    # Weights shape: (out_channels, in_channels/groups, K, K)
    # We need weights for c_out channels and c_in channels
    # Reshape weights access: w[c_out, c_in, k_h, k_w]
    # We can load a block of weights
    # w_ptr offset: c_out * (in_channels/groups * K * K) + c_in * (K * K) + k_h * K + k_w
    # This is complex. Better to load weights in a tiled manner or assume small K.
    # Since K is small, we can load weights per kernel element.
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    # Iterate over kernel dimensions
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Input coordinates corresponding to this kernel element
            # h_in = h_out * stride - padding + k_h
            # w_in = w_out * stride - padding + k_w
            h_in = h_out * stride - padding + k_h
            w_in = w_out * stride - padding + k_w

            # Mask for valid input coordinates
            mask_h_in = (h_in >= 0) & (h_in < height_in)
            mask_w_in = (w_in >= 0) & (w_in < width_in)
            mask_in = mask_h_in[:, None, None] & mask_w_in[None, :, None] & mask_c_in[None, None, :]

            # Load input tile
            x_offsets = (block_n * stride_n + 
                         h_in[:, None, None] * stride_h + 
                         w_in[None, :, None] * stride_w + 
                         c_in[None, None, :])
            x = tl.load(x_ptr + x_offsets, mask=mask_in, other=0.0)

            # Load weight element for this k_h, k_w and channel block
            # w shape: (out_channels, in_channels/groups, K, K)
            # We need w[c_out, c_in - c_in_start, k_h, k_w]
            # c_out is block_c + c_in_group_range
            # c_in is c_in_start + c_in_range
            # w_offsets: (block_c + c_out_group) * (in_channels/groups * K * K) + (c_in_range) * (K * K) + k_h * K + k_w
            # This is still complex.
            # Alternative: Load all weights for the group?
            # Weights for group g: shape (out_channels/groups, in_channels/groups, K, K)
            # We can load this into shared memory or just load directly.
            # Since K is small, we can load w[c_out, c_in, k_h, k_w] directly.
            # w_ptr offset calculation:
            # w_ptr + c_out * (in_channels/groups * K * K) + (c_in - c_in_start) * (K * K) + k_h * K + k_w
            # c_out = block_c + c_out_group
            # c_in - c_in_start = c_in_range
            w_offsets = ((block_c + c_out_group) * (in_channels // groups * kernel_size * kernel_size) + 
                         c_in_range * (kernel_size * kernel_size) + 
                         k_h * kernel_size + k_w)
            w = tl.load(w_ptr + w_offsets, mask=mask_c & mask_c_in, other=0.0)
            
            acc += x * w

    # Add bias
    if b_ptr is not None:
        b_offsets = block_c + c_out_group
        b = tl.load(b_ptr + b_offsets, mask=mask_c, other=0.0)
        acc += b

    # Store output
    y_offsets = (block_n * stride_n + 
                 h_out[:, None, None] * stride_h + 
                 w_out[None, :, None] * stride_w + 
                 c_out[None, None, :])
    tl.store(y_ptr + y_offsets, acc, mask=mask_out)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    groups_per_channel = groups

    height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding

    y = torch.empty((batch_size, out_channels, height_out, width_out), device=x.device, dtype=x.dtype)

    # Grid dimensions
    grid_h = (height_out + 31) // 32
    grid_w = (width_out + 31) // 32
    grid_c = (out_channels + 31) // 32
    grid_n = batch_size

    # Kernel configuration
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_K = kernel_size

    # Launch kernel
    conv_transpose2d_kernel[(grid_n, grid_c, grid_h, grid_w)](
        x, weight, bias, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        kernel_size, stride, padding, output_padding,
        groups, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride[0]
        padding = self.conv_transpose2d.padding[0]
        output_padding = self.conv_transpose2d.output_padding[0]
        groups = self.conv_transpose2d.groups
        
        return triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups)