import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    X,  # Input tensor: (B, C_in, D_in, H_in, W_in)
    W,  # Weight tensor: (C_in, C_out // groups, K_d, K_h, K_w)
    B,  # Bias tensor: (C_out,) or None
    Y,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    batch_size, in_channels, out_channels, groups,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    BLOCK_SIZE_M: tl.constexpr,  # Blocks of output channels
    BLOCK_SIZE_N: tl.constexpr,  # Blocks of spatial elements
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_channel_block = tl.program_id(1)
    pid_spatial_block = tl.program_id(2)
    
    # Calculate output channel range for this block
    out_channel_start = pid_channel_block * BLOCK_SIZE_M
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
    out_channel_mask = out_channel_offsets < out_channels
    
    # Calculate spatial position
    total_spatial = depth_out * height_out * width_out
    spatial_idx = pid_spatial_block * BLOCK_SIZE_N
    spatial_offsets = spatial_idx + tl.arange(0, BLOCK_SIZE_N)
    spatial_mask = spatial_offsets < total_spatial
    
    # Convert linear spatial index to 3D coordinates
    z = spatial_offsets // (height_out * width_out)
    rem = spatial_offsets % (height_out * width_out)
    y = rem // width_out
    x = rem % width_out
    
    # Compute input spatial coordinates (accounting for stride and padding)
    in_z_start = z * stride_d - padding_d
    in_y_start = y * stride_h - padding_h
    in_x_start = x * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(in_channels):
        # Get input channel index
        c_in_offset = c_in
        
        # Compute effective input coordinates
        for kd in range(kernel_d):
            in_z = in_z_start + kd
            if in_z < 0 or in_z >= depth_in:
                continue
                
            for kh in range(kernel_h):
                in_y = in_y_start + kh
                if in_y < 0 or in_y >= height_in:
                    continue
                    
                for kw in range(kernel_w):
                    in_x = in_x_start + kw
                    if in_x < 0 or in_x >= width_in:
                        continue
                    
                    # Compute weight index: W[c_in, c_out, kd, kh, kw]
                    # For groups: weight layout is (C_in, C_out // groups, K_d, K_h, K_w)
                    weight_ptr = W + c_in_offset * (out_channels // groups * kernel_d * kernel_h * kernel_w)
                    weight_ptr += (out_channel_offsets // (out_channels // groups)) * (kernel_d * kernel_h * kernel_w)
                    weight_ptr += (kd * kernel_h * kernel_w + kh * kernel_w + kw)
                    
                    # Compute input index: X[b, c_in, z, y, x]
                    input_ptr = X + pid_batch * (in_channels * depth_in * height_in * width_in)
                    input_ptr += c_in_offset * (depth_in * height_in * width_in)
                    input_ptr += (in_z * height_in * width_in + in_y * width_in + in_x)
                    
                    # Load values
                    w_val = tl.load(weight_ptr, mask=out_channel_mask, other=0.0)
                    x_val = tl.load(input_ptr)
                    
                    # Accumulate
                    acc += w_val * x_val
    
    # Add bias if present
    if B is not None:
        bias_ptr = B + out_channel_offsets
        acc += tl.load(bias_ptr, mask=out_channel_mask, other=0.0)
    
    # Store result
    output_ptr = Y + pid_batch * (out_channels * depth_out * height_out * width_out)
    output_ptr += out_channel_offsets * (depth_out * height_out * width_out)
    output_ptr += z * (height_out * width_out) + y * width_out + x
    
    tl.store(output_ptr, acc, mask=out_channel_mask & spatial_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1
) -> torch.Tensor:
    """Triton implementation of ConvTranspose3d"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    # Weight shape: (in_channels, out_channels // groups, k_d, k_h, k_w)
    _, out_channels, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)
    
    depth_out = (depth_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    height_out = (height_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    width_out = (width_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, device=x.device, dtype=x.dtype)
    
    # Configure grid and block sizes
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 256  # Spatial elements per block
    
    grid = (
        batch_size,
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (depth_out * height_out * width_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, groups,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 3D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )


import math