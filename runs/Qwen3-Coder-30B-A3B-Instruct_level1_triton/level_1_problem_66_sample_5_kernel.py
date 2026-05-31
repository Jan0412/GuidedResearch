import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output dimensions
    group_size = out_channels // groups
    
    # Each thread handles one output element
    if batch_idx >= batch_size or out_ch_idx >= out_channels or out_d_idx >= output_depth or out_h_idx >= output_height or out_w_idx >= output_width:
        return
    
    # Calculate which group this channel belongs to
    group_idx = out_ch_idx // group_size
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input positions with stride and padding
                    input_d = out_d_idx * stride_d - padding_d + kd * dilation_d
                    input_h = out_h_idx * stride_h - padding_h + kh * dilation_h
                    input_w = out_w_idx * stride_w - padding_w + kw * dilation_w
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and 
                        input_h >= 0 and input_h < input_height and 
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input and weight indices
                        input_idx = (batch_idx * in_channels * input_depth * input_height * input_width +
                                   ic * input_depth * input_height * input_width +
                                   input_d * input_height * input_width +
                                   input_h * input_width +
                                   input_w)
                        
                        weight_idx = (out_ch_idx * in_channels * kernel_d * kernel_h * kernel_w +
                                    ic * kernel_d * kernel_h * kernel_w +
                                    kd * kernel_h * kernel_w +
                                    kh * kernel_w +
                                    kw)
                        
                        # Load values and accumulate
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Write output
    output_idx = (batch_idx * out_channels * output_depth * output_height * output_width +
                 out_ch_idx * output_depth * output_height * output_width +
                 out_d_idx * output_height * output_width +
                 out_h_idx * output_width +
                 out_w_idx)
    
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1):
    """
    Triton implementation of 3D convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_d - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_h - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_w - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size and grid
    BLOCK_SIZE = 1024
    GRID_SIZE = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    grid = lambda meta: GRID_SIZE
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Call kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=1
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use the custom Triton implementation
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

# Helper functions for getting inputs
def get_inputs():
    batch_size = 8
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    depth = 16
    height = 128
    width = 128
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]