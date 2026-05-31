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
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channel_id = tl.program_id(2)
    
    # Calculate output indices
    output_idx = tl.program_id(3) * OUTPUT_ELEMENTS_PER_BLOCK
    if output_idx >= output_depth * output_height * output_width:
        return
        
    # Calculate output coordinates
    out_z = output_idx // (output_height * output_width)
    out_y = (output_idx % (output_height * output_width)) // output_width
    out_x = output_idx % output_width
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Handle bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + channel_id, mask=channel_id < out_channels, other=0.0)
    
    # Group handling
    group_start = group_id * GROUPS_PER_BLOCK
    group_end = min(group_start + GROUPS_PER_BLOCK, groups)
    
    # Channel handling
    channel_start = channel_id * CHANNELS_PER_BLOCK
    channel_end = min(channel_start + CHANNELS_PER_BLOCK, in_channels)
    
    # Loop over groups and channels
    for g in range(group_start, group_end):
        for c in range(channel_start, channel_end):
            # Calculate input coordinates for this kernel position
            for kd in range(kernel_depth):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        # Calculate input coordinates
                        input_z = out_z * stride_d + kd - padding_d
                        input_y = out_y * stride_h + kh - padding_h
                        input_x = out_x * stride_w + kw - padding_w
                        
                        # Check bounds
                        if (input_z >= 0 and input_z < input_depth and
                            input_y >= 0 and input_y < input_height and
                            input_x >= 0 and input_x < input_width):
                            
                            # Calculate input index
                            input_idx = (batch_id * (in_channels * input_depth * input_height * input_width) +
                                       (g * (in_channels // groups) + c) * (input_depth * input_height * input_width) +
                                       input_z * (input_height * input_width) +
                                       input_y * input_width +
                                       input_x)
                            
                            # Calculate weight index
                            weight_idx = (g * (out_channels // groups) + channel_id) * (kernel_depth * kernel_height * kernel_width) + \
                                       kd * (kernel_height * kernel_width) + \
                                       kh * kernel_width + \
                                       kw
                            
                            # Load input and weight
                            input_val = tl.load(input_ptr + input_idx, mask=True, other=0.0)
                            weight_val = tl.load(weight_ptr + weight_idx, mask=True, other=0.0)
                            
                            # Accumulate
                            acc += input_val * weight_val
    
    # Store output
    output_idx_total = (batch_id * (out_channels * output_depth * output_height * output_width) +
                       channel_id * (output_depth * output_height * output_width) +
                       out_z * (output_height * output_width) +
                       out_y * output_width +
                       out_x)
    
    tl.store(output_ptr + output_idx_total, acc, mask=output_idx < output_depth * output_height * output_width)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    # Ensure input tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    GROUPS_PER_BLOCK = 1
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 64
    
    # Grid configuration
    grid_batch = batch_size
    grid_groups = math.ceil(groups / GROUPS_PER_BLOCK)
    grid_channels = math.ceil(out_channels / CHANNELS_PER_BLOCK)
    grid_output = math.ceil(output_depth * output_height * output_width / OUTPUT_ELEMENTS_PER_BLOCK)
    
    grid = (grid_batch, grid_groups, grid_channels, grid_output)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Uses custom Triton kernels for optimization.
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
        Performs the transposed 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

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

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )