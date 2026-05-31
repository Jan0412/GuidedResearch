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
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output position
    output_d = output_idx // (output_height * output_width)
    output_h = (output_idx % (output_height * output_width)) // output_width
    output_w = output_idx % output_width
    
    # Calculate input position
    input_d = output_d * stride - padding
    input_h = output_h * stride - padding
    input_w = output_w * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate through kernel
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Calculate input coordinates
                d = input_d + k_d * dilation
                h = input_h + k_h * dilation
                w = input_w + k_w * dilation
                
                # Check bounds
                if d >= 0 and d < input_depth and h >= 0 and h < input_height and w >= 0 and w < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * input_depth * input_height * input_width) +
                                       channel_idx * (input_depth * input_height * input_width) +
                                       d * (input_height * input_width) +
                                       h * input_width +
                                       w)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        channel_idx * (out_channels * kernel_size * kernel_size * kernel_size) +
                                        output_idx % out_channels * (kernel_size * kernel_size * kernel_size) +
                                        k_d * (kernel_size * kernel_size) +
                                        k_h * kernel_size +
                                        k_w)
                    
                    acc += input_val * weight_val
    
    # Store result
    if output_idx < output_depth * output_height * output_width:
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_depth * output_height * output_width) +
                (output_idx % out_channels) * (output_depth * output_height * output_width) +
                output_d * (output_height * output_width) +
                output_h * output_width +
                output_w, acc[0])

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,
        out_channels,
        output_depth * output_height * output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    conv_transpose3d_kernel[grid](
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
        stride,
        padding,
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square kernel, so only one value needed).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None}'