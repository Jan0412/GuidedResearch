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
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_d,
    kernel_h,
    kernel_w,
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
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_depth_idx = tl.program_id(2)
    
    # Calculate output dimensions
    group_size = out_channels // groups
    
    # Each thread block handles one output channel
    if out_channel_idx >= out_channels:
        return
    
    # Calculate which group this channel belongs to
    group_idx = out_channel_idx // group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels (group-wise)
    for ic in range(in_channels // groups):
        # Calculate input channel index within group
        input_channel_idx = group_idx * (in_channels // groups) + ic
        
        # Loop over kernel spatial dimensions
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input positions
                    input_d = out_depth_idx * stride_d + kd * dilation_d - padding_d
                    input_h = 0  # For simplicity, assuming height stride is 1
                    input_w = 0  # For simplicity, assuming width stride is 1
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and 
                        input_h >= 0 and input_h < input_height and 
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input index
                        input_idx = (
                            batch_idx * (in_channels * input_depth * input_height * input_width) +
                            input_channel_idx * (input_depth * input_height * input_width) +
                            input_d * (input_height * input_width) +
                            input_h * input_width +
                            input_w
                        )
                        
                        # Calculate weight index
                        weight_idx = (
                            out_channel_idx * (in_channels // groups * kernel_d * kernel_h * kernel_w) +
                            ic * (kernel_d * kernel_h * kernel_w) +
                            kd * (kernel_h * kernel_w) +
                            kh * kernel_w +
                            kw
                        )
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if exists
    if bias_ptr is not None:
        bias_idx = out_channel_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Write output
    if batch_idx < batch_size and out_depth_idx < output_depth:
        output_idx = (
            batch_idx * (out_channels * output_depth * output_height * output_width) +
            out_channel_idx * (output_depth * output_height * output_width) +
            out_depth_idx * (output_height * output_width)
        )
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_d - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_h - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_w - 1) + 1)) // stride[2] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Launch kernel
    grid = (
        batch_size,
        out_channels,
        output_depth
    )
    
    # Kernel parameters
    BLOCK_SIZE = 128
    GROUP_SIZE = 8
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_d,
        kernel_h,
        kernel_w,
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
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

# For compatibility with original API
def get_inputs():
    batch_size = 8
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    depth = 16
    height = 128
    width = 128
    
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]