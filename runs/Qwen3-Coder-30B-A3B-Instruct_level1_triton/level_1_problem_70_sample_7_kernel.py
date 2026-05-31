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
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output position
    out_d = out_d_idx
    out_h = out_h_idx
    out_w = out_w_idx
    
    # Calculate input position
    input_d = out_d * stride_d - padding_d
    input_h = out_h * stride_h - padding_h
    input_w = out_w * stride_w - padding_w
    
    # Group handling
    group_idx = out_c_idx // (out_channels // groups)
    group_offset = group_idx * (in_channels // groups)
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr + batch_idx * in_channels * input_depth * input_height * input_width, 
                                (in_channels // groups) * input_depth * input_height * input_width, 
                                (in_channels // groups) * input_depth * input_height * input_width)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input indices
                id = input_d + kd * dilation_d
                ih = input_h + kh * dilation_h
                iw = input_w + kw * dilation_w
                
                # Check bounds
                if (id >= 0 and id < input_depth and 
                    ih >= 0 and ih < input_height and 
                    iw >= 0 and iw < input_width):
                    
                    # Calculate weight index
                    weight_idx = out_c_idx * (in_channels // groups) * kernel_depth * kernel_height * kernel_width + \
                                (group_offset + (kd * kernel_height * kernel_width + kh * kernel_width + kw))
                    
                    # Calculate input index
                    input_idx = batch_idx * in_channels * input_depth * input_height * input_width + \
                               (group_offset + (id * input_height * input_width + ih * input_width + iw))
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=(id >= 0 and id < input_depth and 
                                                                    ih >= 0 and ih < input_height and 
                                                                    iw >= 0 and iw < input_width))
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = batch_idx * out_channels * output_depth * output_height * output_width + \
                out_c_idx * output_depth * output_height * output_width + \
                out_d * output_height * output_width + \
                out_h * output_width + \
                out_w
    
    tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), 
                           output_padding=(0,0,0), dilation=(1,1,1), groups=1):
    """
    Triton implementation of ConvTranspose3d
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + 1 + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + 1 + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + 1 + output_padding[2]
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    GROUPS = groups
    
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
        GROUPS=GROUPS
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
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
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
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
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )

# For compatibility with original interface
def get_inputs():
    batch_size = 8
    in_channels = 48
    out_channels = 24
    kernel_size = 3
    depth = 96
    height = 96
    width = 96
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [48, 24, 3]  # in_channels, out_channels, kernel_size