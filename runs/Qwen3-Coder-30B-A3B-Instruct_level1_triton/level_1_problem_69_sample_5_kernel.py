import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_HEIGHT_PER_BLOCK: tl.constexpr,
    OUTPUT_WIDTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_ch_id = tl.program_id(2)
    
    # Calculate output dimensions per block
    output_h_start = tl.program_id(3) * OUTPUT_HEIGHT_PER_BLOCK
    output_w_start = tl.program_id(4) * OUTPUT_WIDTH_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(input_ptr + batch_id * in_channels * input_height * input_width + 
                                  group_id * (in_channels // groups) * input_height * input_width,
                                  [OUTPUT_HEIGHT_PER_BLOCK, OUTPUT_WIDTH_PER_BLOCK])
    
    # Loop over channels
    for ch in range(0, in_channels // groups, CHANNELS_PER_BLOCK):
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_HEIGHT_PER_BLOCK, OUTPUT_WIDTH_PER_BLOCK), dtype=tl.float32)
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input positions
                ih = output_h_start * stride_h - padding_h + kh * dilation_h
                iw = output_w_start * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + batch_id * in_channels * input_height * input_width +
                                       group_id * (in_channels // groups) * input_height * input_width +
                                       ch * input_height * input_width + ih * input_width + iw)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + out_ch_id * groups * in_channels * kernel_height * kernel_width +
                                        group_id * in_channels * kernel_height * kernel_width +
                                        ch * kernel_height * kernel_width + kh * kernel_width + kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
        
        # Store result
        for oh in range(OUTPUT_HEIGHT_PER_BLOCK):
            for ow in range(OUTPUT_WIDTH_PER_BLOCK):
                if output_h_start + oh < output_height and output_w_start + ow < output_width:
                    output_idx = batch_id * out_channels * output_height * output_width + \
                                out_ch_id * output_height * output_width + \
                                (output_h_start + oh) * output_width + (output_w_start + ow)
                    tl.atomic_add(output_ptr + output_idx, acc[oh, ow])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1 + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1 + output_padding[1]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 16
    OUTPUT_HEIGHT_PER_BLOCK = 8
    OUTPUT_WIDTH_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,           # Batch dimension
        groups,               # Groups dimension  
        out_channels,         # Output channels dimension
        (output_height + OUTPUT_HEIGHT_PER_BLOCK - 1) // OUTPUT_HEIGHT_PER_BLOCK,  # Height blocks
        (output_width + OUTPUT_WIDTH_PER_BLOCK - 1) // OUTPUT_WIDTH_PER_BLOCK      # Width blocks
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_HEIGHT_PER_BLOCK=OUTPUT_HEIGHT_PER_BLOCK,
        OUTPUT_WIDTH_PER_BLOCK=OUTPUT_WIDTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    Optimized using custom Triton kernels for better performance.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Convert to contiguous for better memory access patterns
        x = x.contiguous()
        
        # Use Triton kernel implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
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
            f'dilation={self.dilation}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])