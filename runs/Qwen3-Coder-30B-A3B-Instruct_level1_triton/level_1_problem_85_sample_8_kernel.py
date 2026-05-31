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
    channel_block = tl.program_id(1)
    height_block = tl.program_id(2)
    width_block = tl.program_id(3)
    
    # Calculate starting positions for this block
    start_c = channel_block * CHANNELS_PER_BLOCK
    start_h = height_block * HEIGHT_PER_BLOCK
    start_w = width_block * WIDTH_PER_BLOCK
    
    # Shared memory for input tile and weight tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding_h, WIDTH_PER_BLOCK + 2 * padding_w))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(kernel_height, kernel_width))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Calculate input position considering dilation and padding
            input_h = start_h * stride_h + kh * dilation_h - padding_h
            input_w = start_w * stride_w + kw * dilation_w - padding_w
            
            # Load weight
            weight_val = tl.load(weight_ptr + (start_c + kh * kernel_width + kw), mask=(kh < kernel_height) & (kw < kernel_width))
            
            # Load input with boundary checks
            input_val = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
            if (input_h >= 0) and (input_h < height) and (input_w >= 0) and (input_w < width):
                input_val = tl.load(input_ptr + batch_idx * (in_channels * height * width) + 
                                   start_c * (height * width) + 
                                   input_h * width + input_w, 
                                   mask=((tl.arange(0, HEIGHT_PER_BLOCK)[:, None] + input_h) < height) &
                                        ((tl.arange(0, WIDTH_PER_BLOCK)[None, :] + input_w) < width))
            
            # Accumulate
            acc += weight_val * input_val
    
    # Store output
    output_offset = batch_idx * (in_channels * out_height * out_width) + start_c * (out_height * out_width) + start_h * out_width + start_w
    output_mask = ((tl.arange(0, HEIGHT_PER_BLOCK)[:, None] + start_h) < out_height) & \
                  ((tl.arange(0, WIDTH_PER_BLOCK)[None, :] + start_w) < out_width)
    tl.store(output_ptr + output_offset, acc, mask=output_mask)

def triton_depthwise_conv2d(input_tensor, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 16
    HEIGHT_PER_BLOCK = 16
    WIDTH_PER_BLOCK = 16
    
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
    Optimized with Triton kernels.
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
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        # Use Triton implementation
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