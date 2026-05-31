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
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Each thread block processes one output element
    if output_idx >= output_elements:
        return
        
    # Convert linear output index to 3D coordinates
    out_d = output_idx // (output_height * output_width)
    remaining = output_idx % (output_height * output_width)
    out_h = remaining // output_width
    out_w = remaining % output_width
    
    # Calculate input coordinates with padding and stride
    in_d_start = out_d * stride_d - padding_d
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for c in range(in_channels):
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input positions
                    in_d = in_d_start + kd * dilation_d
                    in_h = in_h_start + kh * dilation_h
                    in_w = in_w_start + kw * dilation_w
                    
                    # Check bounds
                    if (in_d >= 0 and in_d < input_depth and 
                        in_h >= 0 and in_h < input_height and 
                        in_w >= 0 and in_w < input_width):
                        
                        # Calculate input index
                        input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                                   c * (input_depth * input_height * input_width) + \
                                   in_d * (input_height * input_width) + \
                                   in_h * input_width + \
                                   in_w
                        
                        # Calculate weight index
                        weight_idx = channel_idx * (in_channels * kernel_depth * kernel_height * kernel_width) + \
                                    c * (kernel_depth * kernel_height * kernel_width) + \
                                    kd * (kernel_height * kernel_width) + \
                                    kh * kernel_width + \
                                    kw
                        
                        # Load input and weight values
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store output
    output_idx_total = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                      channel_idx * (output_depth * output_height * output_width) + \
                      out_d * (output_height * output_width) + \
                      out_h * output_width + \
                      out_w
    
    tl.store(output_ptr + output_idx_total, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define kernel parameters
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 16
    OUTPUT_ELEMENTS_PER_BLOCK = 64
    
    # Calculate grid dimensions
    output_elements = output_depth * output_height * output_width
    grid = (
        batch_size,  # batch dimension
        out_channels,  # channel dimension
        output_elements  # output element dimension
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )

# For compatibility with original API
def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    width = 64
    height = 64
    depth = 64
    
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [3, 64, (3, 5, 7)]