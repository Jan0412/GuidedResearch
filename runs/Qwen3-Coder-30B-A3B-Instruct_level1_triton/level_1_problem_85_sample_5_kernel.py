import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    kernel_height,
    kernel_width,
    output_height,
    output_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_h_id = tl.program_id(2)
    output_w_id = tl.program_id(3)
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * height * width
    channel_offset = channel_id * height * width
    
    # Shared memory for input tile and weight tile
    input_tile = tl.shared_ptr(input_ptr + batch_offset + channel_offset, 
                              shape=[HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK], 
                              dtype=tl.float32)
    weight_tile = tl.shared_ptr(weight_ptr + channel_id * kernel_height * kernel_width, 
                               shape=[kernel_height, kernel_width], 
                               dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = output_h_id * stride_h + kh * dilation_h - padding_h
            iw = output_w_id * stride_w + kw * dilation_w - padding_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + batch_offset + channel_offset + ih * width + iw, 
                                  mask=(ih < height) & (iw < width), other=0.0)
                # Load weight value
                weight_val = tl.load(weight_ptr + channel_id * kernel_height * kernel_width + kh * kernel_width + kw)
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    if output_h_id < output_height and output_w_id < output_width:
        output_idx = batch_id * in_channels * output_height * output_width + \
                    channel_id * output_height * output_width + \
                    output_h_id * output_width + output_w_id
        tl.store(output_ptr + output_idx, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    output_height = (height + 2 * padding_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
    output_width = (width + 2 * padding_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        in_channels,
        output_height,
        output_width
    )
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height,
        width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.stride_h, 
            self.stride_w, 
            self.padding_h, 
            self.padding_w, 
            self.dilation_h, 
            self.dilation_w
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
            
        return output