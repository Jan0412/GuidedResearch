import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Calculate output width index
    out_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate group size
    ch_per_group = in_channels // groups
    
    # Calculate which group this output channel belongs to
    group_idx = out_ch_idx // ch_per_group
    
    # Calculate output position
    out_h = out_h_idx * stride_h - padding_h
    out_w = out_w_idx * stride_w - padding_w
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            in_h = out_h + kh
            in_w = out_w + kw
            
            # Check bounds
            if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                # Calculate input index
                input_idx = batch_idx * (in_channels * input_height * input_width) + \
                           (group_idx * ch_per_group) * (input_height * input_width) + \
                           in_h * input_width + in_w
                
                # Calculate weight index
                weight_idx = out_ch_idx * (ch_per_group * kernel_height * kernel_width) + \
                            (kh * kernel_width + kw)
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + out_ch_idx, mask=True)
        acc += bias_val
    
    # Calculate output index
    output_idx = batch_idx * (out_channels * output_height * output_width) + \
                out_ch_idx * (output_height * output_width) + \
                out_h_idx * output_width + out_w_idx
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d operation
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare grid
    grid = (
        batch_size,
        out_channels,
        (output_height + stride_h - 1) // stride_h,
        (output_width + stride_w - 1) // stride_w
    )
    
    # Launch kernel
    BLOCK_SIZE = 16
    GROUP_SIZE = 8
    
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])