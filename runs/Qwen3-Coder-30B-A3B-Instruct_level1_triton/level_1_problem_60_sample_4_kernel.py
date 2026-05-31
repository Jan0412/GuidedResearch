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
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(1, 1, 1, 1))
    
    # Calculate output position
    out_d = out_d_idx * stride_d - padding_d
    out_h = out_h_idx * stride_h - padding_h
    out_w = out_w_idx * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                in_d = out_d + kd * dilation_d
                in_h = out_h + kh * dilation_h
                in_w = out_w + kw * dilation_w
                
                # Check bounds
                if (in_d >= 0 and in_d < input_depth and 
                    in_h >= 0 and in_h < input_height and 
                    in_w >= 0 and in_w < input_width):
                    
                    # Load input value
                    input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                               in_c_idx * (input_depth * input_height * input_width) + \
                               in_d * (input_height * input_width) + \
                               in_h * input_width + \
                               in_w
                    
                    # Load weight value
                    weight_idx = out_c_idx * (in_channels * kernel_depth * kernel_height * kernel_width) + \
                                in_c_idx * (kernel_depth * kernel_height * kernel_width) + \
                                kd * (kernel_height * kernel_width) + \
                                kh * kernel_width + \
                                kw
                    
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    acc += input_val * weight_val
    
    # Store output
    if batch_idx < batch_size and out_c_idx < out_channels and out_d_idx < output_depth and out_h_idx < output_height and out_w_idx < output_width:
        output_idx = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                    out_c_idx * (output_depth * output_height * output_width) + \
                    out_d_idx * (output_height * output_width) + \
                    out_h_idx * output_width + \
                    out_w_idx
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Custom Triton implementation of 3D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Launch kernel
    conv3d_kernel[grid](
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
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.
    Optimized using custom Triton kernels.
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
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Ensure proper initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using custom Triton kernel.
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'dilation={self.dilation}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])