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
    CHANNELS_PER_GROUP: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    output_element_idx = tl.program_id(2) * OUTPUT_ELEMENTS_PER_BLOCK + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
    
    # Calculate total output elements
    total_output_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    # Early exit if out of bounds
    mask = output_element_idx < total_output_elements
    
    # Calculate which output element this thread handles
    elem_idx = output_element_idx
    out_c = elem_idx % out_channels
    elem_idx //= out_channels
    out_w = elem_idx % output_width
    elem_idx //= output_width
    out_h = elem_idx % output_height
    elem_idx //= output_height
    out_d = elem_idx % output_depth
    elem_idx //= output_depth
    batch = elem_idx
    
    # Check bounds
    if not mask.all():
        return
    
    # Calculate corresponding input coordinates
    # For transposed conv, we compute input positions that contribute to output position
    # Input position = (output_position - padding) / stride
    input_d = out_d * stride_d - padding_d
    input_h = out_h * stride_h - padding_h
    input_w = out_w * stride_w - padding_w
    
    # Group mapping
    group_size = out_channels // groups
    group_id = out_c // group_size
    local_c = out_c % group_size
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate through kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                id = input_d + kd
                ih = input_h + kh
                iw = input_w + kw
                
                # Check bounds for input
                valid_input = (id >= 0) & (id < input_depth) & \
                              (ih >= 0) & (ih < input_height) & \
                              (iw >= 0) & (iw < input_width)
                
                if valid_input:
                    # Calculate input index
                    input_idx = batch * (in_channels * input_depth * input_height * input_width) + \
                               (group_idx * CHANNELS_PER_GROUP + local_c) * (input_depth * input_height * input_width) + \
                               id * (input_height * input_width) + \
                               ih * input_width + \
                               iw
                    
                    # Calculate weight index
                    weight_idx = group_id * (CHANNELS_PER_GROUP * kernel_depth * kernel_height * kernel_width) + \
                                local_c * (kernel_depth * kernel_height * kernel_width) + \
                                kd * (kernel_height * kernel_width) + \
                                kh * kernel_width + \
                                kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = batch * (out_channels * output_depth * output_height * output_width) + \
                 out_c * (output_depth * output_height * output_width) + \
                 out_d * (output_height * output_width) + \
                 out_h * output_width + \
                 out_w
    
    tl.store(output_ptr + output_idx, acc, mask=mask)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    # Input shape: (B, C_in, D_in, H_in, W_in)
    # Weight shape: (C_in, C_out, K_d, K_h, K_w) for grouped version
    # Output shape: (B, C_out, D_out, H_out, W_out)
    
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Constants
    BLOCK_SIZE = 128
    CHANNELS_PER_GROUP = in_channels // groups
    OUTPUT_ELEMENTS_PER_BLOCK = 64
    
    # Grid configuration
    grid_batch = batch_size
    grid_groups = groups
    grid_elements = (batch_size * out_channels * output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    # Launch kernel
    conv_transpose3d_kernel[(grid_batch, grid_groups, grid_elements), 
                           (BLOCK_SIZE, CHANNELS_PER_GROUP, OUTPUT_ELEMENTS_PER_BLOCK)](
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
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
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
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            groups=self.groups
        )