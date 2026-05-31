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
    input_height,
    input_width,
    input_depth,
    kernel_height,
    kernel_width,
    kernel_depth,
    output_height,
    output_width,
    output_depth,
    stride_h,
    stride_w,
    stride_d,
    padding_h,
    padding_w,
    padding_d,
    dilation_h,
    dilation_w,
    dilation_d,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    out_z = tl.program_id(4)
    
    # Calculate output dimensions
    if out_y >= output_height or out_x >= output_width or out_z >= output_depth:
        return
        
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group handling
    group_size = out_channels // groups
    group_id = out_channel_idx // group_size
    
    # Loop over kernel spatial dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for kd in range(kernel_depth):
                # Calculate input coordinates
                input_y = out_y * stride_h + kh * dilation_h - padding_h
                input_x = out_x * stride_w + kw * dilation_w - padding_w
                input_z = out_z * stride_d + kd * dilation_d - padding_d
                
                # Check bounds
                if (input_y >= 0 and input_y < input_height and 
                    input_x >= 0 and input_x < input_width and 
                    input_z >= 0 and input_z < input_depth):
                    
                    # Load input value
                    input_offset = (batch_idx * in_channels * input_height * input_width * input_depth + 
                                  group_id * (input_height * input_width * input_depth) +
                                  input_y * (input_width * input_depth) + 
                                  input_x * input_depth + 
                                  input_z)
                    input_val = tl.load(input_ptr + input_offset, mask=True)
                    
                    # Load weight value
                    weight_offset = (out_channel_idx * in_channels * kernel_height * kernel_width * kernel_depth + 
                                   group_id * (in_channels * kernel_height * kernel_width * kernel_depth) +
                                   kh * (kernel_width * kernel_depth) + 
                                   kw * kernel_depth + 
                                   kd)
                    weight_val = tl.load(weight_ptr + weight_offset, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Write output
    output_offset = (batch_idx * out_channels * output_height * output_width * output_depth + 
                    out_channel_idx * output_height * output_width * output_depth + 
                    out_y * (output_width * output_depth) + 
                    out_x * output_depth + 
                    out_z)
    tl.store(output_ptr + output_offset, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_height, input_width, input_depth = input_tensor.shape
    out_channels, _, kernel_height, kernel_width, kernel_depth = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_depth = (input_depth + 2 * padding[2] - (dilation[2] * (kernel_depth - 1) + 1)) // stride[2] + 1
    
    # Ensure contiguous tensors
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, output_depth, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size and grid
    BLOCK_SIZE = 16
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width,
        output_depth
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        input_depth,
        kernel_height,
        kernel_width,
        kernel_depth,
        output_height,
        output_width,
        output_depth,
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
        GROUP_SIZE=1
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    Optimized with Triton kernels for better performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, 1) if isinstance(stride, int) else (stride[0], stride[1], 1)
        self.padding = (padding, padding, 0) if isinstance(padding, int) else (padding[0], padding[1], 0)
        self.dilation = (dilation, dilation, 1) if isinstance(dilation, int) else (dilation[0], dilation[1], 1)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
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

# Make sure to handle the specific case of the kernel being 3D but with depth=1
# We can optimize further by treating this as a 2D convolution in the spatial dimensions
# but keeping the third dimension separate for performance