import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    weight_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Shared memory for input tile
    TILE_H = 16
    TILE_W = 16
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over kernel
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Calculate input positions
            ih = tl.arange(0, TILE_H) * stride_h - padding_h + kh * dilation_h
            iw = tl.arange(0, TILE_W) * stride_w - padding_w + kw * dilation_w
            
            # Bounds checking for input
            ih_bound = (ih >= 0) & (ih < input_height)
            iw_bound = (iw >= 0) & (iw < input_width)
            
            # Load input values
            input_vals = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
            for c in range(0, in_channels):
                # Load input chunk
                input_chunk = tl.load(input_ptr + 
                    pid_batch * input_stride_0 + 
                    c * input_stride_1 + 
                    tl.broadcast_to(ih[:, None], (TILE_H, TILE_W)) * input_stride_2 +
                    tl.broadcast_to(iw[None, :], (TILE_H, TILE_W)) * input_stride_3,
                    mask=(ih_bound[:, None] & iw_bound[None, :]), other=0.0)
                # Load weight
                weight_val = tl.load(weight_ptr + 
                    pid_out_ch * weight_stride_0 + 
                    c * weight_stride_1 + 
                    kh * weight_stride_2 + 
                    kw * weight_stride_3)
                acc += input_chunk * weight_val
    
    # Apply bias if exists
    if has_bias:
        bias_val = tl.load(bias_ptr + pid_out_ch)
        acc += bias_val
    
    # Store output
    output_offset = pid_batch * output_stride_0 + pid_out_ch * output_stride_1
    for i in range(0, TILE_H):
        for j in range(0, TILE_W):
            if i < output_height and j < output_width:
                tl.store(output_ptr + output_offset + i * output_stride_2 + j * output_stride_3, acc[i, j])

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Set up grid
    grid = (
        batch_size,
        out_channels
    )
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Call kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Extract parameters
        stride = self.conv2d.stride if isinstance(self.conv2d.stride, tuple) else (self.conv2d.stride, self.conv2d.stride)
        padding = self.conv2d.padding if isinstance(self.conv2d.padding, tuple) else (self.conv2d.padding, self.conv2d.padding)
        dilation = self.conv2d.dilation if isinstance(self.conv2d.dilation, tuple) else (self.conv2d.dilation, self.conv2d.dilation)
        
        # Use Triton kernel
        return triton_conv2d(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride,
            padding,
            dilation,
            self.conv2d.groups
        )

# Note: This implementation has limitations compared to PyTorch's native implementation,
# particularly regarding memory usage and handling of edge cases. For production use,
# a more sophisticated approach would be needed with proper tiling and shared memory management.