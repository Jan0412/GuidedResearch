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
    group_size,
    BLOCK_SIZE: tl.constexpr,
    THREADS_PER_GROUP: tl.constexpr
):
    # Get thread and block indices
    block_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate which output element this block handles
    batch_idx = block_id // (output_depth * output_height * output_width)
    remaining = block_id % (output_depth * output_height * output_width)
    out_d = remaining // (output_height * output_width)
    remaining = remaining % (output_height * output_width)
    out_h = remaining // output_width
    out_w = remaining % output_width
    
    # Shared memory for input tile
    shared_input = tl.shared_pointer(input_ptr + batch_idx * in_channels * input_depth * input_height * input_width)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate input position
                in_d = out_d * stride_depth - padding_depth + k_d
                in_h = out_h * stride_height - padding_height + k_h
                in_w = out_w * stride_width - padding_width + k_w
                
                # Check bounds
                if (in_d >= 0 and in_d < input_depth and 
                    in_h >= 0 and in_h < input_height and 
                    in_w >= 0 and in_w < input_width):
                    
                    # Calculate input index
                    input_idx = (batch_idx * in_channels * input_depth * input_height * input_width + 
                                group_id * group_size * input_depth * input_height * input_width + 
                                in_d * input_height * input_width + 
                                in_h * input_width + 
                                in_w)
                    
                    # Calculate weight index
                    weight_idx = (group_id * group_size * out_channels * kernel_depth * kernel_height * kernel_width + 
                                 k_d * kernel_height * kernel_width + 
                                 k_h * kernel_width + 
                                 k_w)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = (batch_idx * out_channels * output_depth * output_height * output_width + 
                 group_id * output_depth * output_height * output_width + 
                 out_d * output_height * output_width + 
                 out_h * output_width + 
                 out_w)
    
    tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_depth, stride_height, stride_width = stride
    padding_depth, padding_height, padding_width = padding
    output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Calculate grid
    total_elements = batch_size * output_depth * output_height * output_width
    num_blocks = total_elements
    num_groups = groups
    
    # Grid configuration
    grid = (num_blocks, num_groups)
    BLOCK_SIZE = 128
    
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
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        groups,
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE,
        THREADS_PER_GROUP=32
    )
    
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.
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

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )