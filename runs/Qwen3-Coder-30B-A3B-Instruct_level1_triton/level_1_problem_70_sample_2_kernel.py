import torch
import torch.nn as nn
import torch.nn.functional as F
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
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channel_id = tl.program_id(2)
    
    # Calculate group info
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input indices
                d_in = batch_id * stride_d + kd * dilation_d - padding_d
                h_in = channel_id * stride_h + kh * dilation_h - padding_h
                w_in = group_id * stride_w + kw * dilation_w - padding_w
                
                # Check bounds
                if (d_in >= 0 and d_in < input_depth and 
                    h_in >= 0 and h_in < input_height and 
                    w_in >= 0 and w_in < input_width):
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_id * in_channels * input_depth * input_height * input_width +
                                       channel_id * input_depth * input_height * input_width +
                                       d_in * input_height * input_width +
                                       h_in * input_width +
                                       w_in)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        group_id * out_channels_per_group * channels_per_group * kernel_depth * kernel_height * kernel_width +
                                        channel_id * kernel_depth * kernel_height * kernel_width +
                                        kd * kernel_height * kernel_width +
                                        kh * kernel_width +
                                        kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store output
    if batch_id < batch_size and channel_id < out_channels:
        output_idx = batch_id * out_channels * output_depth * output_height * output_width + \
                    channel_id * output_depth * output_height * output_width
        tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, dilation, groups):
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define kernel parameters
    BLOCK_SIZE = 16
    GROUPS_PER_BLOCK = 1
    CHANNELS_PER_BLOCK = 1
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use our custom Triton implementation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

# Keep original class for comparison
class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)