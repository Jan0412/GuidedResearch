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
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    
    # Calculate global thread index
    tid = tl.program_id(0) * (tl.num_programs(1) * tl.num_programs(2)) + \
          tl.program_id(1) * tl.num_programs(2) + tl.program_id(2)
    
    # Shared memory for weight tiles
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_PER_BLOCK, CHANNELS_PER_BLOCK, kernel_depth, kernel_height, kernel_width))
    
    # Load weights for this group and channel
    if group_idx < groups and channel_idx < out_channels:
        weight_offset = group_idx * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width + \
                       (channel_idx % (out_channels // groups)) * in_channels * kernel_depth * kernel_height * kernel_width
        for k_d in range(kernel_depth):
            for k_h in range(kernel_height):
                for k_w in range(kernel_width):
                    for c in range(in_channels):
                        if c < in_channels and k_d < kernel_depth and k_h < kernel_height and k_w < kernel_width:
                            shared_weight[group_idx, channel_idx % (out_channels // groups), k_d, k_h, k_w] = \
                                tl.load(weight_ptr + weight_offset + c * kernel_depth * kernel_height * kernel_width + 
                                       k_d * kernel_height * kernel_width + k_h * kernel_width + k_w)
    
    # Process output elements
    for out_d in range(output_depth):
        for out_h in range(output_height):
            for out_w in range(output_width):
                # Calculate input coordinates
                in_d_start = out_d * stride_d - padding_d
                in_h_start = out_h * stride_h - padding_h
                in_w_start = out_w * stride_w - padding_w
                
                # Initialize accumulator
                acc = 0.0
                
                # Convolution computation
                for k_d in range(kernel_depth):
                    for k_h in range(kernel_height):
                        for k_w in range(kernel_width):
                            in_d = in_d_start + k_d
                            in_h = in_h_start + k_h
                            in_w = in_w_start + k_w
                            
                            # Check bounds
                            if (in_d >= 0 and in_d < input_depth and
                                in_h >= 0 and in_h < input_height and
                                in_w >= 0 and in_w < input_width):
                                
                                # Compute input index
                                input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                                           (in_d * input_height * input_width + in_h * input_width + in_w) * in_channels
                                
                                # Compute weight index
                                weight_idx = group_idx * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width + \
                                            (channel_idx % (out_channels // groups)) * in_channels * kernel_depth * kernel_height * kernel_width + \
                                            k_d * kernel_height * kernel_width + k_h * kernel_width + k_w
                                
                                # Accumulate
                                acc += tl.load(input_ptr + input_idx + (channel_idx // (out_channels // groups))) * \
                                       tl.load(weight_ptr + weight_idx)
                
                # Store result
                if batch_idx < batch_size and channel_idx < out_channels:
                    output_idx = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                                (channel_idx * output_depth * output_height * output_width + out_d * output_height * output_width + out_h * output_width + out_w)
                    tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of 3D transposed convolution.
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels
    )
    
    BLOCK_SIZE = 128
    GROUPS_PER_BLOCK = 1
    CHANNELS_PER_BLOCK = 1
    
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
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])