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
    out_height,
    out_width,
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
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate global indices
    global_channel = channel_idx * CHANNELS_PER_BLOCK + tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None]
    global_out_h = out_h_idx * HEIGHT_PER_BLOCK + tl.arange(0, HEIGHT_PER_BLOCK)[None, :, None]
    global_out_w = out_w_idx * WIDTH_PER_BLOCK + tl.arange(0, WIDTH_PER_BLOCK)[None, None, :]
    
    # Create masks for valid channels
    channel_mask = global_channel < in_channels
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            ih = global_out_h * stride_h - padding_h + kh * dilation_h
            iw = global_out_w * stride_w - padding_w + kw * dilation_w
            
            # Create input mask
            input_mask = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
            
            # Load input data
            input_val = tl.load(input_ptr + 
                               batch_idx * in_channels * height * width +
                               global_channel * height * width +
                               ih * width +
                               iw,
                               mask=input_mask & channel_mask, other=0.0)
            
            # Load kernel data
            kernel_val = tl.load(weight_ptr + 
                                global_channel * kernel_height * kernel_width +
                                kh * kernel_width +
                                kw,
                                mask=channel_mask, other=0.0)
            
            # Accumulate
            acc += input_val * kernel_val[:, None, None]
    
    # Write output
    output_mask = (global_out_h < out_height) & (global_out_w < out_width) & channel_mask
    tl.store(output_ptr + 
             batch_idx * in_channels * out_height * out_width +
             global_channel * out_height * out_width +
             global_out_h * out_width +
             global_out_w,
             acc,
             mask=output_mask & channel_mask)

def triton_depthwise_conv2d(input_tensor, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise 2D convolution.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    
    # Get dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 4
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (out_height + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (out_width + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
    )
    
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
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK
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
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
            
        return output