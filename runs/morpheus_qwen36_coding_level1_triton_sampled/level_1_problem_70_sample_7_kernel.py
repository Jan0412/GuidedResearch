import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements_x,
    n_elements_out,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    in_channels,
    out_channels,
    kernel_size,
    depth_in,
    height_in,
    width_in,
    depth_out,
    height_out,
    width_out,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Grid mapping: (N, C_out, D_out)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # Tile offsets
    offsets_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offsets_h = tl.arange(0, BLOCK_SIZE_H)
    offsets_w = tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for valid output elements
    mask_d = offsets_d < depth_out
    mask_h = offsets_h < height_out
    mask_w = offsets_w < width_out
    mask_out = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Base pointers for output
    out_base = pid_n * out_channels * depth_out * height_out * width_out + pid_c * depth_out * height_out * width_out
    out_ptr_tile = out_ptr + out_base
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Compute input coordinates
                    # y[n, c, d, h, w] += x[n, c_in, d - kd, h - kh, w - kw] * w[c_in, c, kd, kh, kw]
                    # Note: ConvTranspose indices are inverted compared to Conv
                    # Output index (d, h, w) maps to input index (d - kd, h - kh, w - kw) for stride=1, padding=0
                    # General formula:
                    # x_d = d + kd * stride - 2 * padding + dilation * kd
                    # Actually, for ConvTranspose:
                    # y[n, c, d, h, w] = sum_{c_in, kd, kh, kw} x[n, c_in, d + kd*stride - padding, ...] * w[c_in, c, kd, kh, kw]
                    # We need to be careful with the sign.
                    # Standard ConvTranspose3d:
                    # out[n, c, d, h, w] = sum_{c_in, kd, kh, kw} in[n, c_in, d + kd*stride - padding, h + kh*stride - padding, w + kw*stride - padding] * weight[c_in, c, kd, kh, kw]
                    
                    # Compute input spatial indices
                    in_d = pid_d * BLOCK_SIZE_D + offsets_d + kd * stride - padding
                    in_h = offsets_h + kh * stride - padding
                    in_w = offsets_w + kw * stride - padding
                    
                    # Mask for valid input indices
                    mask_in_d = (in_d >= 0) & (in_d < depth_in)
                    mask_in_h = (in_h >= 0) & (in_h < height_in)
                    mask_in_w = (in_w >= 0) & (in_w < width_in)
                    mask_in = mask_in_d[:, None, None] & mask_in_h[None, :, None] & mask_in_w[None, None, :]
                    
                    # Load input tile
                    x_offset = pid_n * in_channels * depth_in * height_in * width_in + c_in * depth_in * height_in * width_in
                    x_ptr_tile = x_ptr + x_offset
                    x_val = tl.load(x_ptr_tile + in_d[:, None, None] * height_in * width_in + in_h[None, :, None] * width_in + in_w[None, None, :], mask=mask_in, other=0.0)
                    
                    # Load weight element
                    w_offset = c_in * out_channels * kernel_size * kernel_size * kernel_size + pid_c * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Handle output_padding
    # Output padding adds extra elements to the output.
    # If output_padding > 0, the last few elements might not be covered by the standard loop.
    # We can handle this by checking bounds or adjusting indices.
    # For simplicity, we assume the grid covers the full output including output_padding.
    # The mask_out already handles bounds.
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = pid_c
        b_val = tl.load(b_ptr + b_offset)
        acc += b_val
    
    # Store result
    tl.store(out_ptr_tile + offsets_d[:, None, None] * height_out * width_out + offsets_h[None, :, None] * width_out + offsets_w[None, None, :], acc, mask=mask_out)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    
    # Compute output dimensions
    depth_out = (depth_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # We use a 3D grid for (N, C_out, D_out)
    # Each block handles a tile of (BLOCK_D, BLOCK_H, BLOCK_W)
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    grid = (batch_size, out_channels, depth_out)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        x.numel(), out.numel(),
        stride, padding, output_padding, dilation, groups,
        in_channels, out_channels, kernel_size,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        BLOCK_SIZE_D=BLOCK_D,
        BLOCK_SIZE_H=BLOCK_H,
        BLOCK_SIZE_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride[0]
        padding = self.conv_transpose3d.padding[0]
        output_padding = self.conv_transpose3d.output_padding[0]
        dilation = self.conv_transpose3d.dilation[0]
        groups = self.conv_transpose3d.groups
        
        return triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups)