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
    input_shape,
    weight_shape,
    output_shape,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate output position
    out_pos = batch_idx * out_channels * output_depth * output_height * output_width + \
              out_channel_idx * output_depth * output_height * output_width + \
              out_d * output_height * output_width + \
              out_h * output_width + \
              out_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(in_channels):
        for kd in range(weight_shape[2]):
            for kh in range(weight_shape[3]):
                for kw in range(weight_shape[4]):
                    # Compute input coordinates
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input coordinates are valid
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Compute input position
                        input_pos = batch_idx * in_channels * input_depth * input_height * input_width + \
                                   ic * input_depth * input_height * input_width + \
                                   input_d * input_height * input_width + \
                                   input_h * input_width + \
                                   input_w
                        
                        # Compute weight position
                        weight_pos = out_channel_idx * in_channels * weight_shape[2] * weight_shape[3] * weight_shape[4] + \
                                    ic * weight_shape[2] * weight_shape[3] * weight_shape[4] + \
                                    kd * weight_shape[3] * weight_shape[4] + \
                                    kh * weight_shape[4] + \
                                    kw
                        
                        # Accumulate
                        input_val = tl.load(input_ptr + input_pos, mask=True)
                        weight_val = tl.load(weight_ptr + weight_pos, mask=True)
                        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + out_pos, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D transposed convolution
    """
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get shapes
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_d - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_h - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_w - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.shape,
        weight.shape,
        output.shape,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        BLOCK_SIZE=BLOCK_SIZE
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
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]