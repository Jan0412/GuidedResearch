import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    stride, padding,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid mapping: (b, c_block, h_block, w_block)
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)

    # Calculate start indices for this block
    c_out_start = c_block * BLOCK_C
    h_out_start = h_block * BLOCK_H
    w_out_start = w_block * BLOCK_W

    # Create offsets for the block
    c_off = c_out_start + tl.arange(0, BLOCK_C)
    h_off = h_out_start + tl.arange(0, BLOCK_H)
    w_off = w_out_start + tl.arange(0, BLOCK_W)

    # Masks to handle boundary conditions
    mask_c = c_off < C_out
    mask_h = h_off < H_out
    mask_w = w_off < W_out
    mask_block = mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and kernel elements
    for c_in in tl.range(C_in):
        for kh in tl.range(K_h):
            for kw in tl.range(K_w):
                # Load weights for current block of output channels
                # Weight shape: (C_out, C_in, K_h, K_w)
                w_idx = c_out_start + c_off
                w_ptr_offset = w_idx[:, None, None] * (C_in * K_h * K_w) + c_in * (K_h * K_w) + kh * K_w + kw
                w = tl.load(w_ptr + w_ptr_offset, mask=mask_c[:, None, None], other=0.0)

                # Load input tile
                # Input shape: (B, C_in, H_in, W_in)
                # Output spatial: h_out_start + kh + h_off, w_out_start + kw + w_off
                x_h = h_out_start + kh + h_off
                x_w = w_out_start + kw + w_off
                
                # Bounds check for input loading
                mask_x_h = x_h < H_in
                mask_x_w = x_w < W_in
                mask_x = mask_x_h[None, :, None] & mask_x_w[None, None, :]
                
                x_ptr_offset = b * (C_in * H_in * W_in) + c_in * (H_in * W_in) + x_h[:, None, None] * W_in + x_w[None, :, None]
                x = tl.load(x_ptr + x_ptr_offset, mask=mask_x, other=0.0)

                # Accumulate
                acc += w * x

    # Store result
    out_ptr_offset = b * (C_out * H_out * W_out) + c_out_start[:, None, None] * (H_out * W_out) + h_out_start[None, :, None] * W_out + w_out_start[None, None, :]
    tl.store(out_ptr + out_ptr_offset, acc, mask=mask_block)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    B, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = w.shape
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1

    assert x.is_contiguous() and w.is_contiguous()
    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_C = 8
    BLOCK_H = 64
    BLOCK_W = 64

    grid = (B, triton.cdiv(C_out, BLOCK_C), triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))

    conv2d_kernel[grid](
        x, w, out,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        1, 0,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.conv2d.weight)