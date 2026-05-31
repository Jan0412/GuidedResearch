import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    output_idx = tl.program_id(2) * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Loop over output elements
    for i in range(OUTPUT_ELEMENTS_PER_BLOCK):
        if output_idx + i >= output_elements:
            break
            
        # Calculate output coordinates
        out_z = (output_idx + i) // (output_width * output_height)
        out_y = ((output_idx + i) % (output_width * output_height)) // output_width
        out_x = (output_idx + i) % output_width
        
        # Calculate input coordinates with padding
        in_z_start = out_z * stride_d - padding_d
        in_y_start = out_y * stride_h - padding_h
        in_x_start = out_x * stride_w - padding_w
        
        # Accumulate result
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over input channels and kernel
        for c in range(in_channels):
            # Loop over kernel dimensions
            for kd in range(kernel_depth):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        # Calculate input position
                        in_z = in_z_start + kd * dilation_d
                        in_y = in_y_start + kh * dilation_h
                        in_x = in_x_start + kw * dilation_w
                        
                        # Check bounds
                        if (in_z >= 0 and in_z < input_depth and 
                            in_y >= 0 and in_y < input_height and 
                            in_x >= 0 and in_x < input_width):
                            
                            # Load input value
                            input_val = tl.load(input_ptr + 
                                batch_id * (in_channels * input_depth * input_height * input_width) +
                                c * (input_depth * input_height * input_width) +
                                in_z * (input_height * input_width) +
                                in_y * input_width +
                                in_x)
                            
                            # Load weight value
                            weight_val = tl.load(weight_ptr + 
                                out_channel_id * (in_channels * kernel_depth * kernel_height * kernel_width) +
                                c * (kernel_depth * kernel_height * kernel_width) +
                                kd * (kernel_height * kernel_width) +
                                kh * kernel_width +
                                kw)
                            
                            acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + 
            batch_id * (out_channels * output_depth * output_height * output_width) +
            out_channel_id * (output_depth * output_height * output_width) +
            out_z * (output_height * output_width) +
            out_y * output_width +
            out_x, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid configuration
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid dimensions
    grid_batch = batch_size
    grid_channels = (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    grid_output = (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    # Launch kernel
    conv3d_kernel[(grid_batch, grid_channels, grid_output),](
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )