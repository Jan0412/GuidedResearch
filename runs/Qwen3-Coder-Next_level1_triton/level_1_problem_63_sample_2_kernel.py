import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, H, W)
    w_ptr,  # Weight tensor (out_channels, in_channels, kH, kW)
    b_ptr,  # Bias tensor (out_channels,)
    y_ptr,  # Output tensor (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    height, width, kernel_size,
    stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_oc = tl.program_id(3)
    
    # Calculate output coordinates
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    out_c_start = pid_oc * BLOCK_OC
    
    # Calculate input coordinates
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Create ranges for output height and width
    offsets_h = tl.arange(0, BLOCK_H)
    offsets_w = tl.arange(0, BLOCK_W)
    offsets_oc = tl.arange(0, BLOCK_OC)
    
    # Create masks for valid output positions
    mask_h = offsets_h < BLOCK_H
    mask_w = offsets_w < BLOCK_W
    mask_oc = out_c_start + offsets_oc < out_channels
    
    # Calculate output tensor offsets
    y_batch_offset = pid_batch * out_channels * height * width
    y_h_offset = offsets_h * width
    y_w_offset = offsets_w
    y_c_offset = (out_c_start + offsets_oc) * height * width
    
    # Initialize accumulator for output
    y = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_OC), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                # Calculate input position
                in_h = in_h_start + kh * dilation
                in_w = in_w_start + kw * dilation
                
                # Check if input position is valid
                mask_in_h = (in_h >= 0) & (in_h < height)
                mask_in_w = (in_w >= 0) & (in_w < width)
                mask_in = mask_in_h[:, None] & mask_in_w[None, :]
                
                # Load input values
                x_offset = y_batch_offset + (in_h[:, None] * width + in_w[None, :]) * in_channels + ic
                x_val = tl.load(x_ptr + x_offset, mask=mask_in & (offsets_h[:, None] < BLOCK_H) & (offsets_w[None, :] < BLOCK_W), other=0.0)
                
                # Load weight values
                w_offset = (out_c_start + offsets_oc) * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)
                
                # Accumulate convolution result
                y += x_val[:, :, None] * w_val[None, None, :]
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_start + offsets_oc, mask=mask_oc, other=0.0)
        y += bias[None, None, :]
    
    # Store output
    y_offset = y_batch_offset + (out_h * width + out_w) * out_channels + out_c_start
    tl.store(y_ptr + y_offset, y, mask=(offsets_h[:, None, None] < BLOCK_H) & (offsets_w[None, :, None] < BLOCK_W) & (offsets_oc[None, None, :] < BLOCK_OC))


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> torch.Tensor:
    """
    Triton implementation of 2D convolution.
    Assumes groups=1 for simplicity (standard convolution).
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (kernel_size_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Set block sizes for optimization
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_OC = 16
    BLOCK_IC = 4
    
    # Grid dimensions
    grid = (
        batch_size,  # batch
        (out_height + BLOCK_H - 1) // BLOCK_H,  # height blocks
        (out_width + BLOCK_W - 1) // BLOCK_W,   # width blocks
        (out_channels + BLOCK_OC - 1) // BLOCK_OC,  # out_channels blocks
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        height, width, kernel_size_h,
        stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
        KERNEL_H=kernel_size_h, KERNEL_W=kernel_size_w,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the convolution model using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the original conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        
        # Use our Triton implementation
        return triton_conv2d(
            x, weight, bias,
            stride=self.conv2d.stride[0],
            padding=self.conv2d.padding[0],
            dilation=self.conv2d.dilation[0],
            groups=self.conv2d.groups
        )