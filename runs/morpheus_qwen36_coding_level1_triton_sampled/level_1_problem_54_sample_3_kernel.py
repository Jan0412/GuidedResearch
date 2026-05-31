import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    n_batch,
    c_in,
    c_out,
    k_size,
    d_in,
    h_in,
    w_in,
    d_out,
    h_out,
    w_out,
    stride,
    padding,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    n = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    d_idx = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    h_idx = tl.program_id(3) * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = tl.program_id(4) * BLOCK_W + tl.arange(0, BLOCK_W)

    out_ptr_offset = n * (c_out * d_out * h_out * w_out) + c_out_idx * (d_out * h_out * w_out) + d_idx * (h_out * w_out) + h_idx * w_out + w_idx

    w_ptr_offset = c_out_idx * (c_in * k_size * k_size * k_size)
    w_offsets = tl.arange(0, c_in)[:, None, None, None] * (k_size * k_size * k_size) + tl.arange(0, k_size)[None, :, None, None] * (k_size * k_size) + tl.arange(0, k_size)[None, None, :, None] * k_size + tl.arange(0, k_size)[None, None, None, :]
    w_mask = (tl.arange(0, c_in)[:, None, None, None] < c_in) & (tl.arange(0, k_size)[None, :, None, None] < k_size) & (tl.arange(0, k_size)[None, None, :, None] < k_size) & (tl.arange(0, k_size)[None, None, None, :] < k_size)
    weights = tl.load(w_ptr + w_ptr_offset + w_offsets, mask=w_mask, other=0.0)

    x_ptr_offset = n * (c_in * d_in * h_in * w_in)
    in_d = d_idx * stride - padding + tl.arange(0, k_size)[None, None, :]
    in_h = h_idx * stride - padding + tl.arange(0, k_size)[None, :, None]
    in_w = w_idx * stride - padding + tl.arange(0, k_size)[:, None, None]

    x_offsets = in_d * (h_in * w_in) + in_h * w_in + in_w + tl.arange(0, c_in)[:, None, None, None] * (d_in * h_in * w_in)
    mask_d = (in_d >= 0) & (in_d < d_in)
    mask_h = (in_h >= 0) & (in_h < h_in)
    mask_w = (in_w >= 0) & (in_w < w_in)
    spatial_mask = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    mask = spatial_mask[None, :, :, :]

    x_patch = tl.load(x_ptr + x_ptr_offset + x_offsets, mask=mask, other=0.0)

    out = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    for k_d in range(k_size):
        for k_h in range(k_size):
            for k_w in range(k_size):
                w_slice = weights[:, k_d, k_h, k_w]
                x_slice = x_patch[:, d_idx + k_d - (d_idx[0] - (d_idx[0] - d_idx[0])), h_idx + k_h - (h_idx[0] - h_idx[0]), w_idx + k_w - (w_idx[0] - w_idx[0])]
                out += tl.sum(w_slice[:, None, None, None] * x_slice, axis=0)

    tl.store(out_ptr + out_ptr_offset, out, mask=(d_idx < d_out) & (h_idx < h_out) & (w_idx < w_out))


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.stride(0) != 1 or x.stride(1) != 1:
            x = x.contiguous()
        
        w = self.conv3d.weight
        if w.stride(0) != 1:
            w = w.contiguous()
            
        n_batch, c_in, d_in, h_in, w_in = x.shape
        c_out = self.conv3d.out_channels
        k_size = self.conv3d.kernel_size[0]
        stride = self.conv3d.stride[0]
        padding = self.conv3d.padding[0]
        
        d_out = (d_in + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
        h_out = (h_in + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1
        
        out = torch.empty((n_batch, c_out, d_out, h_out, w_out), device=x.device, dtype=x.dtype)
        
        BLOCK_D = 4
        BLOCK_H = 4
        BLOCK_W = 8
        
        grid = (n_batch, c_out, (d_out + BLOCK_D - 1) // BLOCK_D, (h_out + BLOCK_H - 1) // BLOCK_H, (w_out + BLOCK_W - 1) // BLOCK_W)
        
        conv3d_kernel[grid](
            x, w, out,
            x.numel(),
            n_batch, c_in, c_out, k_size,
            d_in, h_in, w_in,
            d_out, h_out, w_out,
            stride, padding,
            BLOCK_D, BLOCK_H, BLOCK_W
        )
        return out