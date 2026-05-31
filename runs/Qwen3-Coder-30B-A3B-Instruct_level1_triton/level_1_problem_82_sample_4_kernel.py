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
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    
    # Calculate output coordinates
    out_h_start = height_idx * stride
    out_w_start = width_idx * stride
    
    # Shared memory for input tile and kernel
    shared_input = tl.shared_block_alloc((HEIGHT_PER_BLOCK + 2 * padding, WIDTH_PER_BLOCK + 2 * padding))
    shared_kernel = tl.shared_block_alloc((kernel_size, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load kernel weights
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            if k_h < kernel_size and k_w < kernel_size:
                shared_kernel[k_h, k_w] = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + k_h * kernel_size + k_w)
    
    # Load input tile with padding
    for h in range(HEIGHT_PER_BLOCK + 2 * padding):
        for w in range(WIDTH_PER_BLOCK + 2 * padding):
            if h < HEIGHT_PER_BLOCK + 2 * padding and w < WIDTH_PER_BLOCK + 2 * padding:
                ih = out_h_start + h - padding
                iw = out_w_start + w - padding
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    val = tl.load(input_ptr + 
                                 batch_idx * in_channels * input_height * input_width +
                                 channel_idx * input_height * input_width +
                                 ih * input_width + iw)
                else:
                    val = 0.0
                    
                shared_input[h, w] = val
    
    # Compute convolution
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            if k_h < kernel_size and k_w < kernel_size:
                kernel_val = shared_kernel[k_h, k_w]
                for h in range(HEIGHT_PER_BLOCK):
                    for w in range(WIDTH_PER_BLOCK):
                        if h < HEIGHT_PER_BLOCK and w < WIDTH_PER_BLOCK:
                            input_val = shared_input[k_h + h, k_w + w]
                            acc += input_val * kernel_val
    
    # Write output
    if height_idx < output_height and width_idx < output_width:
        output_offset = batch_idx * in_channels * output_height * output_width + \
                       channel_idx * output_height * output_width + \
                       height_idx * output_width + width_idx
        tl.store(output_ptr + output_offset, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    grid = (
        batch_size,
        in_channels,
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
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Optimized with custom Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Use Triton kernel for computation
        output = triton_depthwise_conv2d(x, self.weight, self.stride, self.padding)
        
        # Add bias if present
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output