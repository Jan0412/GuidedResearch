import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer
    output_ptr,    # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    dilation_d,
    dilation_w,
    dilation_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    
    # Calculate output dimensions
    out_w_idx = tl.program_id(3)
    out_h_idx = tl.program_id(4)
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=(GROUP_SIZE, 1, 1, 1), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate base positions
    input_d_base = out_d_idx * stride_d - padding_d
    input_w_base = out_w_idx * stride_w - padding_w
    input_h_base = out_h_idx * stride_h - padding_h
    
    # Group handling
    group_size = in_channels // groups
    group_idx = out_c_idx // (out_channels // groups)
    weight_offset = group_idx * group_size * out_channels * kernel_depth * kernel_width * kernel_height
    input_offset = batch_idx * in_channels * input_depth * input_width * input_height
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kw in range(kernel_width):
            for kh in range(kernel_height):
                # Calculate input position
                d = input_d_base + kd * dilation_d
                w = input_w_base + kw * dilation_w
                h = input_h_base + kh * dilation_h
                
                # Check bounds
                if d >= 0 and d < input_depth and w >= 0 and w < input_width and h >= 0 and h < input_height:
                    # Calculate input index
                    input_idx = input_offset + group_idx * group_size * input_depth * input_width * input_height + \
                                d * input_width * input_height + w * input_height + h
                    
                    # Calculate weight index
                    weight_idx = weight_offset + (out_c_idx % (out_channels // groups)) * kernel_depth * kernel_width * kernel_height + \
                                 kd * kernel_width * kernel_height + kw * kernel_height + kh
                    
                    # Load values and accumulate
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Store result
    output_idx = batch_idx * out_channels * output_depth * output_width * output_height + \
                 out_c_idx * output_depth * output_width * output_height + \
                 out_d_idx * output_width * output_height + \
                 out_w_idx * output_height + \
                 out_h_idx
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Custom Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_width,
        output_height
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
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
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
        BLOCK_SIZE=1024,
        GROUP_SIZE=32
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized 3D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
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
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )