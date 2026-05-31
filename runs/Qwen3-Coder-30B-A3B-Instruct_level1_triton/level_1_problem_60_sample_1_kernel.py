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
    # Get block IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_id = tl.program_id(2)
    
    # Calculate output indices
    output_idx = output_id * OUTPUT_ELEMENTS_PER_BLOCK + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
    
    # Early exit if out of bounds
    if tl.any(output_idx >= output_depth * output_height * output_width):
        return
        
    # Calculate which output position this block handles
    out_d = output_idx // (output_height * output_width)
    out_h = (output_idx % (output_height * output_width)) // output_width
    out_w = output_idx % output_width
    
    # Filter out-of-bounds indices
    valid_mask = (out_d < output_depth) & (out_h < output_height) & (out_w < output_width)
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates with stride and padding
                in_d = out_d * stride_d - padding_d + kd * dilation_d
                in_h = out_h * stride_h - padding_h + kh * dilation_h
                in_w = out_w * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                in_bounds = (in_d >= 0) & (in_d < input_depth) & \
                           (in_h >= 0) & (in_h < input_height) & \
                           (in_w >= 0) & (in_w < input_width)
                
                # Load input data
                input_val = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
                if in_bounds:
                    input_idx = batch_id * (in_channels * input_depth * input_height * input_width) + \
                               channel_id * (input_depth * input_height * input_width) + \
                               in_d * (input_height * input_width) + \
                               in_h * input_width + \
                               in_w
                    input_val = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)
                
                # Load weight
                weight_idx = channel_id * (out_channels * kernel_depth * kernel_height * kernel_width) + \
                            (out_channels * kernel_depth * kernel_height * kernel_width) * (kd * kernel_height * kernel_width + kh * kernel_width + kw)
                weight_val = tl.load(weight_ptr + weight_idx, mask=valid_mask, other=0.0)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Write output
    output_idx_full = batch_id * (out_channels * output_depth * output_height * output_width) + \
                     channel_id * (output_depth * output_height * output_width) + \
                     out_d * (output_height * output_width) + \
                     out_h * output_width + \
                     out_w
    tl.store(output_ptr + output_idx_full, acc, mask=valid_mask)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Custom Triton implementation of 3D convolution.
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Calculate grid dimensions
    grid_batch = batch_size
    grid_channels = out_channels
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
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if present
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization