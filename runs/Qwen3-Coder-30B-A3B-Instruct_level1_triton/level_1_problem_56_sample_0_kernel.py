import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_group_idx = tl.program_id(1)
    output_row = tl.program_id(2)
    output_col = tl.program_id(3)
    
    # Calculate global output position
    output_idx = batch_idx * out_channels * output_height * output_width + \
                 channel_group_idx * output_height * output_width + \
                 output_row * output_width + output_col
    
    # Shared memory for input tile and weight tile
    shared_input = tl.shared_ptr(input_ptr, shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    shared_weight = tl.shared_ptr(weight_ptr, shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # For each group
    for g in range(groups):
        # Calculate group-specific pointers
        weight_offset = g * out_channels // groups * in_channels // groups * kernel_height * kernel_width
        input_offset = batch_idx * in_channels * input_height * input_width + g * group_size * input_height * input_width
        
        # Calculate kernel positions
        kh_start = 0
        kw_start = 0
        kh_end = kernel_height
        kw_end = kernel_width
        
        # Loop over kernel elements
        for kh in range(kh_start, kh_end):
            for kw in range(kw_start, kw_end):
                # Calculate input position
                ih = output_row * stride_h - padding_h + kh * dilation_h
                iw = output_col * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Calculate input index
                    input_idx = input_offset + ih * input_width + iw
                    
                    # Calculate weight index
                    weight_idx = weight_offset + channel_group_idx * in_channels // groups * kernel_height * kernel_width + \
                                 (kh * kernel_width + kw)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if available
    if bias_ptr is not None:
        bias_idx = channel_group_idx
        acc += tl.load(bias_ptr + bias_idx)
    
    # Store result
    tl.store(output_ptr + output_idx, acc)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D convolution.
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channel groups
        output_height,        # output rows
        output_width          # output columns
    )
    
    # Launch kernel
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Optimized using custom Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )

# For backward compatibility with original interface
Model = ModelNew