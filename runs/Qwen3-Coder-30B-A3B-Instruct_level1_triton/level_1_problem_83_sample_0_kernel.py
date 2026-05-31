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
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_block = tl.program_id(1)
    height_block = tl.program_id(2)
    width_block = tl.program_id(3)
    
    # Calculate global indices
    channel_start = channel_block * CHANNELS_PER_BLOCK
    height_start = height_block * HEIGHT_PER_BLOCK
    width_start = width_block * WIDTH_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding_h, WIDTH_PER_BLOCK + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel height and width
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            ih = height_start * stride_h + kh * dilation_h - padding_h
            iw = width_start * stride_w + kw * dilation_w - padding_w
            
            # Load input tile with boundary checking
            for h in range(HEIGHT_PER_BLOCK):
                for w in range(WIDTH_PER_BLOCK):
                    if ih + h >= 0 and ih + h < height and iw + w >= 0 and iw + w < width:
                        shared_input[h + padding_h][w + padding_w] = tl.load(
                            input_ptr + 
                            batch_idx * (in_channels * height * width) +
                            channel_start * (height * width) +
                            (ih + h) * width +
                            (iw + w),
                            mask=((ih + h) < height) & ((iw + w) < width)
                        )
                    else:
                        shared_input[h + padding_h][w + padding_w] = 0.0
            
            # Load weight
            weight_val = tl.load(weight_ptr + channel_start * kernel_height * kernel_width + kh * kernel_width + kw)
            
            # Compute partial products
            for h in range(HEIGHT_PER_BLOCK):
                for w in range(WIDTH_PER_BLOCK):
                    acc[h][w] += shared_input[h + padding_h][w + padding_w] * weight_val
    
    # Store output
    for h in range(HEIGHT_PER_BLOCK):
        for w in range(WIDTH_PER_BLOCK):
            if height_start + h < output_height and width_start + w < output_width:
                tl.store(
                    output_ptr + 
                    batch_idx * (in_channels * output_height * output_width) +
                    channel_start * (output_height * output_width) +
                    (height_start + h) * output_width +
                    (width_start + w),
                    acc[h][w]
                )

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise convolution for 2D input with asymmetric kernel.
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    output_height = (height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (output_height + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (output_width + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
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
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
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
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, 1), 
            padding=(self.padding, 0), 
            dilation=(self.dilation, 1)
        )