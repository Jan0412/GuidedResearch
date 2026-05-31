import torch
import torch.nn as nn
import torch.nn.functional as F
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
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_block = tl.program_id(2)
    
    # Calculate starting positions
    start_c = channel_block * CHANNELS_PER_BLOCK
    end_c = tl.minimum(start_c + CHANNELS_PER_BLOCK, out_channels)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Loop over output spatial dimensions
    for out_d in range(output_depth):
        for out_h in range(output_height):
            for out_w in range(output_width):
                # Calculate corresponding input positions
                in_d_start = out_d * stride_d - padding_d
                in_h_start = out_h * stride_h - padding_h
                in_w_start = out_w * stride_w - padding_w
                
                # Initialize accumulator
                acc = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)
                
                # Loop over kernel dimensions
                for k_d in range(kernel_depth):
                    for k_h in range(kernel_height):
                        for k_w in range(kernel_width):
                            # Calculate input coordinates
                            in_d = in_d_start + k_d
                            in_h = in_h_start + k_h
                            in_w = in_w_start + k_w
                            
                            # Check bounds
                            if (in_d >= 0 and in_d < input_depth and 
                                in_h >= 0 and in_h < input_height and 
                                in_w >= 0 and in_w < input_width):
                                
                                # Load input value
                                input_offset = (batch_idx * in_channels * input_depth * input_height * input_width + 
                                              group_idx * (in_channels // groups) * input_depth * input_height * input_width + 
                                              in_d * input_height * input_width + 
                                              in_h * input_width + 
                                              in_w)
                                
                                input_val = tl.load(input_ptr + input_offset, mask=True)
                                
                                # Load weight value
                                weight_offset = (group_idx * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width + 
                                               (start_c - group_idx * (out_channels // groups)) * in_channels * kernel_depth * kernel_height * kernel_width + 
                                               k_d * kernel_height * kernel_width * in_channels + 
                                               k_h * kernel_width * in_channels + 
                                               k_w * in_channels + 
                                               0)  # Simplified indexing
                                
                                weight_val = tl.load(weight_ptr + weight_offset, mask=True)
                                
                                # Accumulate
                                acc += input_val * weight_val
                
                # Write output
                for c in range(start_c, end_c):
                    output_offset = (batch_idx * out_channels * output_depth * output_height * output_width + 
                                   c * output_depth * output_height * output_width + 
                                   out_d * output_height * output_width + 
                                   out_h * output_width + 
                                   out_w)
                    
                    tl.store(output_ptr + output_offset, acc[c - start_c], mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1):
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
    CHANNELS_PER_BLOCK = 8
    OUTPUT_ELEMENTS_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,          # batch dimension
        groups,              # groups dimension  
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK  # channel blocks
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
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel using Triton optimization.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
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

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use Triton implementation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )