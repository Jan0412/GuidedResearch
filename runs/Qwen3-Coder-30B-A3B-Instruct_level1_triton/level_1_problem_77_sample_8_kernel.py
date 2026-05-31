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
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    
    # Calculate output position
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for ic in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input position
                    input_d = out_d_idx * stride - padding + kd * dilation
                    input_h = out_h_idx * stride - padding + kh * dilation
                    input_w = out_w_idx * stride - padding + kw * dilation
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input index
                        input_idx = (
                            batch_idx * (in_channels * input_depth * input_height * input_width) +
                            ic * (input_depth * input_height * input_width) +
                            input_d * (input_height * input_width) +
                            input_h * input_width +
                            input_w
                        )
                        
                        # Calculate weight index
                        weight_idx = (
                            out_ch_idx * (in_channels * kernel_size * kernel_size * kernel_size) +
                            ic * (kernel_size * kernel_size * kernel_size) +
                            kd * (kernel_size * kernel_size) +
                            kh * kernel_size +
                            kw
                        )
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_idx)
                        weight_val = tl.load(weight_ptr + weight_idx)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Write result
    if acc != 0.0:
        output_idx = (
            batch_idx * (out_channels * output_depth * output_height * output_width) +
            out_ch_idx * (output_depth * output_height * output_width) +
            out_d_idx * (output_height * output_width) +
            out_h_idx * output_width +
            out_w_idx
        )
        tl.store(output_ptr + output_idx, acc)

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
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
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
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if present
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
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