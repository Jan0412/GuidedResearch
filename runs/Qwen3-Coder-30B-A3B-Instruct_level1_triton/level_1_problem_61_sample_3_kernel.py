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
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate output position
    out_pos = batch_idx * out_channels * output_depth * output_height * output_width + \
              out_ch_idx * output_depth * output_height * output_width + \
              out_d * output_height * output_width + \
              out_h * output_width + \
              out_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    input_d = out_d * stride_d - padding_d + kd
                    input_h = out_h * stride_h - padding_h + kh
                    input_w = out_w * stride_w - padding_w + kw
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input position
                        input_pos = batch_idx * in_channels * input_depth * input_height * input_width + \
                                   ic * input_depth * input_height * input_width + \
                                   input_d * input_height * input_width + \
                                   input_h * input_width + \
                                   input_w
                        
                        # Calculate weight position
                        weight_pos = out_ch_idx * in_channels * kernel_depth * kernel_height * kernel_width + \
                                    ic * kernel_depth * kernel_height * kernel_width + \
                                    kd * kernel_height * kernel_width + \
                                    kh * kernel_width + \
                                    kw
                        
                        # Load values and accumulate
                        input_val = tl.load(input_ptr + input_pos, mask=True)
                        weight_val = tl.load(weight_ptr + weight_pos, mask=True)
                        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + out_pos, acc[0], mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0)):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    # Get shapes
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare pointers
    input_ptr = input_tensor.data_ptr()
    weight_ptr = weight.data_ptr()
    output_ptr = output.data_ptr()
    
    # Configure grid
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    OUTPUT_BLOCK_SIZE = 1
    
    conv_transpose3d_kernel[grid](
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
        padding_d,
        padding_h,
        padding_w,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_BLOCK_SIZE=OUTPUT_BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Normalize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use Triton kernel for computation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding
        )

# Keep original class for comparison
class Model(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)