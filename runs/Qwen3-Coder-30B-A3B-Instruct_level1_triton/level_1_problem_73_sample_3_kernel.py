import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
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
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate output position
    out_pos = batch_idx * out_channels * output_depth * output_height * output_width + \
              group_idx * channels_per_group * output_depth * output_height * output_width + \
              out_d * output_height * output_width + \
              out_h * output_width + \
              out_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate input positions for this output position
    input_d_start = out_d * stride_d - pad_d
    input_h_start = out_h * stride_h - pad_h
    input_w_start = out_w * stride_w - pad_w
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                input_d = input_d_start + kd
                input_h = input_h_start + kh
                input_w = input_w_start + kw
                
                # Check if input coordinate is valid
                if input_d >= 0 and input_d < input_depth and \
                   input_h >= 0 and input_h < input_height and \
                   input_w >= 0 and input_w < input_width:
                    
                    # Calculate input position
                    input_pos = batch_idx * in_channels * input_depth * input_height * input_width + \
                               group_idx * channels_per_group * input_depth * input_height * input_width + \
                               input_d * input_height * input_width + \
                               input_h * input_width + \
                               input_w
                    
                    # Calculate weight position
                    weight_pos = group_idx * channels_per_group * kernel_depth * kernel_height * kernel_width + \
                                0 * kernel_depth * kernel_height * kernel_width + \
                                kd * kernel_height * kernel_width + \
                                kh * kernel_width + \
                                kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_pos, mask=True)
                    weight_val = tl.load(weight_ptr + weight_pos, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_pos = group_idx * channels_per_group + 0
        bias_val = tl.load(bias_ptr + bias_pos, mask=True)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + out_pos, acc, mask=True)

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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Set kernel attributes
        self.kernel_depth = kernel_size
        self.kernel_height = kernel_size
        self.kernel_width = kernel_size
        
        # Set stride attributes
        self.stride_d = stride
        self.stride_h = stride
        self.stride_w = stride
        
        # Set padding attributes
        self.pad_d = padding
        self.pad_h = padding
        self.pad_w = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride_d - 2 * self.pad_d + self.kernel_depth
        output_height = (input_height - 1) * self.stride_h - 2 * self.pad_h + self.kernel_height
        output_width = (input_width - 1) * self.stride_w - 2 * self.pad_w + self.kernel_width
        
        # Ensure proper alignment for Triton
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Prepare bias pointer
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Define block sizes
        BLOCK_SIZE = 1024
        GROUPS_PER_BLOCK = 4
        CHANNELS_PER_BLOCK = 32
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            self.groups,
            output_depth,
            output_height,
            output_width
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x,
            weight,
            output,
            bias_ptr,
            batch_size,
            in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_depth,
            self.kernel_height,
            self.kernel_width,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.pad_d,
            self.pad_h,
            self.pad_w,
            self.groups,
            in_channels // self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
        
        return output

# Keep original class for reference
class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)