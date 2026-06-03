import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr, w_ptr, out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size,
    out_depth, out_height, out_width,
    # Strides
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_out_c, w_stride_in_c, w_stride_kd, w_stride_kh, w_stride_kw,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    # Convolution parameters
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_d: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_d: tl.constexpr, dil_h: tl.constexpr, dil_w: tl.constexpr,
    # Block sizes
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs for output tensor
    pid_n = tl.program_id(0)  # batch index
    pid_out_c = tl.program_id(1)  # output channel index
    
    # Calculate starting positions for output volume
    start_d = tl.program_id(2) * BLOCK_D
    start_h = tl.program_id(3) * BLOCK_H
    start_w = tl.program_id(4) * BLOCK_W
    
    # Create ranges for output volume
    depth_offsets = start_d + tl.arange(0, BLOCK_D)
    height_offsets = start_h + tl.arange(0, BLOCK_H)
    width_offsets = start_w + tl.arange(0, BLOCK_W)
    
    # Create masks for valid indices
    depth_mask = depth_offsets < out_depth
    height_mask = height_offsets < out_height
    width_mask = width_offsets < out_width
    
    # Broadcast masks for 3D volume
    depth_mask_3d = depth_mask[:, None, None]
    height_mask_3d = height_mask[None, :, None]
    width_mask_3d = width_mask[None, None, :]
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution loop over input channels and kernel dimensions
    for in_c in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input position
                    d_in = (depth_offsets * stride_d - pad_d + kd * dil_d)
                    h_in = (height_offsets * stride_h - pad_h + kh * dil_h)
                    w_in = (width_offsets * stride_w - pad_w + kw * dil_w)
                    
                    # Create masks for input position
                    d_mask = (d_in >= 0) & (d_in < depth)
                    h_mask = (h_in >= 0) & (h_in < height)
                    w_mask = (w_in >= 0) & (w_in < width)
                    
                    # Combine masks
                    input_mask = depth_mask_3d & height_mask_3d & width_mask_3d & \
                                (d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])
                    
                    # Load input values
                    d_indices = d_in * x_stride_d
                    h_indices = h_in * x_stride_h
                    w_indices = w_in * x_stride_w
                    x_offset = pid_n * x_stride_n + in_c * x_stride_c + d_indices[:, None, None] + h_indices[None, :, None] + w_indices[None, None, :]
                    x_vals = tl.load(x_ptr + x_offset, mask=input_mask, other=0.0)
                    
                    # Load weight values
                    w_offset = pid_out_c * w_stride_out_c + in_c * w_stride_in_c + \
                              kd * w_stride_kd + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate convolution result
                    output += x_vals * w_val
    
    # Store result
    out_d_indices = depth_offsets * out_stride_d
    out_h_indices = height_offsets * out_stride_h
    out_w_indices = width_offsets * out_stride_w
    out_offset = pid_n * out_stride_n + pid_out_c * out_stride_c + out_d_indices[:, None, None] + out_h_indices[None, :, None] + out_w_indices[None, None, :]
    tl.store(out_ptr + out_offset, output.to(x_ptr.dtype.element_ty), mask=depth_mask_3d & height_mask_3d & width_mask_3d)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Performs 3D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    
    # Calculate output dimensions
    out_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, 
                     dtype=x.dtype, device=x.device)
    
    # Define block sizes for parallelization
    BLOCK_C = 1  # Input channels processed per kernel call
    BLOCK_K = 1  # Kernel size dimension (we loop over this)
    BLOCK_D = min(4, out_depth)  # Depth block size
    BLOCK_H = min(4, out_height)  # Height block size
    BLOCK_W = min(4, out_width)  # Width block size
    
    # Calculate grid dimensions
    grid = (batch_size, out_channels, 
            (out_depth + BLOCK_D - 1) // BLOCK_D,
            (out_height + BLOCK_H - 1) // BLOCK_H,
            (out_width + BLOCK_W - 1) // BLOCK_W)
    
    # Calculate strides
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w = x.stride()
    w_stride_out_c, w_stride_in_c, w_stride_kd, w_stride_kh, w_stride_kw = weight.stride()
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size,
        out_depth, out_height, out_width,
        x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
        w_stride_out_c, w_stride_in_c, w_stride_kd, w_stride_kh, w_stride_kw,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        stride, stride, stride,  # stride_d, stride_h, stride_w
        padding, padding, padding,  # pad_d, pad_h, pad_w
        dilation, dilation, dilation,  # dil_d, dil_h, dil_w
        BLOCK_C, BLOCK_K, BLOCK_D, BLOCK_H, BLOCK_W
    )
    
    # Add bias if provided
    if bias is not None:
        bias_view = bias.view(1, -1, 1, 1, 1)
        out = out + bias_view
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model with Triton-based 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Register weights as buffers so they persist through the module
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights similar to nn.Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        return triton_conv3d(x, self.weight, self.bias, 
                            stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups)

import math