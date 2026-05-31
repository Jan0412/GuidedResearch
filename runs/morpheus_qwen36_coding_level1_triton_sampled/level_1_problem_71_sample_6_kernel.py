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
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    bias,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Calculate global output coordinates
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_h_block = tl.program_id(2)
    out_w_block = tl.program_id(3)
    
    # Calculate output coordinates for this thread
    ty = tl.program_id(4)
    tx = tl.program_id(5)
    
    h_out = out_h_block * BLOCK_H + ty
    w_out = out_w_block * BLOCK_W + tx
    
    # Check bounds
    if h_out >= height_out or w_out >= width_out:
        return
    
    # Calculate input coordinates relative to output
    # ConvTranspose2d is equivalent to Conv2d with flipped kernel and padding = kernel_size - 1
    # For Conv2d: y[h, w] = sum_{dy, dx} x[h - padding + dy, w - padding + dx] * w[dy, dx]
    # Here padding = kernel_size - 1
    pad = kernel_size - 1
    
    # Input tile start coordinates
    h_in_start = h_out - pad
    w_in_start = w_out - pad
    
    # Load input tile and weights, compute output
    acc = 0.0
    
    # Channels per group
    c_in_per_group = in_channels // groups
    c_out_per_group = out_channels // groups
    
    # Output channel index
    c_out = group_id * c_out_per_group + ty
    
    if c_out >= out_channels:
        return
        
    # Input channel loop
    for c_in_off in range(0, c_in_per_group, BLOCK_C):
        c_in = group_id * c_in_per_group + c_in_off + tl.arange(0, BLOCK_C)
        mask_c = c_in < in_channels
        
        # Input tile offsets
        h_in = tl.arange(0, BLOCK_H) + h_in_start
        w_in = tl.arange(0, BLOCK_W) + w_in_start
        
        # Load input tile
        x_offsets = h_in[:, None] * width_in + w_in[None, :]
        x_mask = (h_in[:, None] >= 0) & (h_in[:, None] < height_in) & \
                 (w_in[None, :] >= 0) & (w_in[None, :] < width_in)
        
        x = tl.load(x_ptr + batch_id * in_channels * height_in * width_in + 
                    c_in[:, None, None] * height_in * width_in + x_offsets,
                    mask=x_mask[:, None, None], other=0.0)
        
        # Load weight tile
        # Weights shape: (out_channels, in_channels // groups, kernel_size, kernel_size)
        # We need weights for c_out and c_in
        dy = tl.arange(0, kernel_size)
        dx = tl.arange(0, kernel_size)
        
        w_offsets = c_out * c_in_per_group * kernel_size * kernel_size + \
                    c_in[:, None, None] * kernel_size * kernel_size + \
                    dy[:, None, None] * kernel_size + dx[None, None, :]
        
        w = tl.load(w_ptr + w_offsets, mask=mask_c[:, None, None], other=0.0)
        
        # Compute partial sum
        acc += tl.sum(x * w, axis=0)
    
    # Add bias if present
    if bias:
        acc += tl.load(b_ptr + c_out)
    
    # Store output
    y_offsets = batch_id * out_channels * height_out * width_out + \
                c_out * height_out * width_out + \
                h_out * width_out + w_out
    tl.store(y_ptr + y_offsets, acc)


def triton_conv_transpose2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                            in_channels: int, out_channels: int, kernel_size: int,
                            stride: int, padding: int, output_padding: int, groups: int, bias: bool):
    batch_size, c_in, h_in, w_in = x.shape
    c_out, c_in_w, k, k_w = w.shape
    
    assert c_in_w == c_in // groups, "Weight channels mismatch"
    assert k == k_w, "Kernel must be square"
    
    # Calculate output dimensions
    h_out = (h_in - 1) * stride - 2 * padding + kernel_size + output_padding
    w_out = (w_in - 1) * stride - 2 * padding + kernel_size + output_padding
    
    y = torch.empty((batch_size, out_channels, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Grid dimensions
    grid = (batch_size, groups, 
            (h_out + 15) // 16, (w_out + 15) // 16)
    
    # Block dimensions
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, w, b, y,
        batch_size, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        kernel_size, stride, padding, output_padding, groups, bias,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias if self.bias else None
        
        # ConvTranspose2d is equivalent to Conv2d with flipped kernel and padding = kernel_size - 1
        w_flipped = torch.flip(w, dims=[2, 3])
        
        return triton_conv_transpose2d(
            x, w_flipped, b,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.kernel_size - 1, self.output_padding, self.groups, self.bias
        )