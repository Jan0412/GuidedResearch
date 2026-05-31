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
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_c_idx = tl.program_id(2)
    
    # Calculate output dimensions per group
    channels_per_group = out_channels // groups
    out_c_start = group_idx * channels_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.float32, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Loop over kernel spatial dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate output position
                od = kd * stride_depth - padding_depth
                oh = kh * stride_height - padding_height
                ow = kw * stride_width - padding_width
                
                # Load weight
                weight_offset = out_c_idx * in_channels * kernel_depth * kernel_height * kernel_width
                weight_offset += (kd * kernel_height * kernel_width + kh * kernel_width + kw) * in_channels
                weight_val = tl.load(weight_ptr + weight_offset + group_idx * channels_per_group * in_channels)
                
                # Loop over output positions
                for out_d in range(output_depth):
                    for out_h in range(output_height):
                        for out_w in range(output_width):
                            # Calculate input position
                            in_d = out_d * stride_depth + kd - padding_depth
                            in_h = out_h * stride_height + kh - padding_height
                            in_w = out_w * stride_width + kw - padding_width
                            
                            # Check bounds
                            if (in_d >= 0 and in_d < input_depth and 
                                in_h >= 0 and in_h < input_height and 
                                in_w >= 0 and in_w < input_width):
                                
                                # Load input value
                                input_offset = batch_idx * in_channels * input_depth * input_height * input_width
                                input_offset += group_idx * (input_depth * input_height * input_width // groups)
                                input_offset += in_d * input_height * input_width + in_h * input_width + in_w
                                input_val = tl.load(input_ptr + input_offset)
                                
                                # Accumulate result
                                output_offset = batch_idx * out_channels * output_depth * output_height * output_width
                                output_offset += out_c_idx * output_depth * output_height * output_width
                                output_offset += out_d * output_height * output_width + out_h * output_width + out_w
                                
                                # Use atomic add to accumulate
                                tl.atomic_add(output_ptr + output_offset, input_val * weight_val)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define kernel parameters
    BLOCK_SIZE = 16
    GROUP_SIZE = 32
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels
    )
    
    # Launch kernel
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
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
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
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )

# For compatibility with existing test code
def get_inputs():
    batch_size = 8
    in_channels = 32
    out_channels = 32
    kernel_size = (3, 5, 7)
    depth = 12
    height = 24
    width = 48
    stride = (2, 2, 2)
    padding = (1, 2, 3)
    output_padding = (1, 1, 1)
    groups = 4
    
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    batch_size = 8
    in_channels = 32
    out_channels = 32
    kernel_size = (3, 5, 7)
    depth = 12
    height = 24
    width = 48
    stride = (2, 2, 2)
    padding = (1, 2, 3)
    output_padding = (1, 1, 1)
    groups = 4
    
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]