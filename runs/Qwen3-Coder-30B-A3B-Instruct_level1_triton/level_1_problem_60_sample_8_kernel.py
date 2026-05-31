import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,   # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    output_ptr,  # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_out_d = tl.program_id(2)
    pid_out_h = tl.program_id(3)
    pid_out_w = tl.program_id(4)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Calculate tile boundaries
    batch_offset = pid_batch * in_channels * input_depth * input_height * input_width
    weight_offset = pid_out_ch * in_channels * kernel_depth * kernel_height * kernel_width
    
    # Loop over K dimension (input channels and kernel elements)
    for k in range(0, in_channels * kernel_depth * kernel_height * kernel_width, BLOCK_SIZE_K):
        # Compute indices for current tile
        k_offset = k
        
        # Load weights (this will be done in chunks)
        weight_tile = tl.load(weight_ptr + weight_offset + k_offset + tl.arange(0, BLOCK_SIZE_K))
        
        # Load input patches
        input_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        
        # Process kernel elements
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    d = pid_out_d * stride_d - pad_d + kd * dilation_d
                    h = pid_out_h * stride_h - pad_h + kh * dilation_h
                    w = pid_out_w * stride_w - pad_w + kw * dilation_w
                    
                    # Check bounds
                    if (d >= 0 and d < input_depth and 
                        h >= 0 and h < input_height and 
                        w >= 0 and w < input_width):
                        
                        # Calculate input index
                        input_idx = batch_offset + (k // (kernel_depth * kernel_height * kernel_width)) * input_depth * input_height * input_width + \
                                   d * input_height * input_width + h * input_width + w
                        
                        # Load input value
                        input_val = tl.load(input_ptr + input_idx)
                        
                        # Accumulate
                        acc += input_val * weight_tile[k % BLOCK_SIZE_K]
        
        # Update accumulator
        acc += tl.sum(input_tile * weight_tile, axis=1)
    
    # Store output
    output_idx = pid_batch * out_channels * output_depth * output_height * output_width + \
                 pid_out_ch * output_depth * output_height * output_width + \
                 pid_out_d * output_height * output_width + \
                 pid_out_h * output_width + \
                 pid_out_w
    
    tl.store(output_ptr + output_idx, acc)

# Simplified fused version focusing on core computation
@triton.jit
def conv3d_fused_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    output_elem = pid
    
    # Convert linear index to 3D coordinates
    out_w = output_elem % output_width
    out_h = (output_elem // output_width) % output_height
    out_d = (output_elem // (output_width * output_height)) % output_depth
    out_ch = (output_elem // (output_width * output_height * output_depth)) % out_channels
    batch = (output_elem // (output_width * output_height * output_depth * out_channels)) % batch_size
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop through kernel and input channels
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for ic in range(in_channels):
                    # Calculate input position
                    d = out_d * stride_d - pad_d + kd * dilation_d
                    h = out_h * stride_h - pad_h + kh * dilation_h
                    w = out_w * stride_w - pad_w + kw * dilation_w
                    
                    # Check bounds
                    if (d >= 0 and d < input_depth and 
                        h >= 0 and h < input_height and 
                        w >= 0 and w < input_width):
                        
                        # Calculate input index
                        input_idx = (batch * in_channels * input_depth * input_height * input_width +
                                   ic * input_depth * input_height * input_width +
                                   d * input_height * input_width +
                                   h * input_width +
                                   w)
                        
                        # Calculate weight index
                        weight_idx = (out_ch * in_channels * kernel_depth * kernel_height * kernel_width +
                                    ic * kernel_depth * kernel_height * kernel_width +
                                    kd * kernel_height * kernel_width +
                                    kh * kernel_width +
                                    kw)
                        
                        # Load values and accumulate
                        input_val = tl.load(input_ptr + input_idx)
                        weight_val = tl.load(weight_ptr + weight_idx)
                        acc += input_val * weight_val
    
    # Store output
    output_idx = (batch * out_channels * output_depth * output_height * output_width +
                 out_ch * output_depth * output_height * output_width +
                 out_d * output_height * output_width +
                 out_h * output_width +
                 out_w)
    
    tl.store(output_ptr + output_idx, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid size
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv3d_fused_kernel[grid_size](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        # Extract parameters
        stride = self.conv3d.stride if isinstance(self.conv3d.stride, tuple) else (self.conv3d.stride, self.conv3d.stride, self.conv3d.stride)
        padding = self.conv3d.padding if isinstance(self.conv3d.padding, tuple) else (self.conv3d.padding, self.conv3d.padding, self.conv3d.padding)
        dilation = self.conv3d.dilation if isinstance(self.conv3d.dilation, tuple) else (self.conv3d.dilation, self.conv3d.dilation, self.conv3d.dilation)
        
        return triton_conv3d(
            x, 
            self.conv3d.weight, 
            self.conv3d.bias,
            stride=stride,
            padding=padding,
            dilation=dilation
        )