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
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_row = tl.program_id(2)
    output_col = tl.program_id(3)
    
    # Calculate global output position
    output_pos = batch_idx * (in_channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_row * output_width + output_col
    
    # Shared memory for input tile and kernel
    input_tile = tl.shared_tensor(tl.arange(0, BLOCK_SIZE), dtype=tl.float32)
    kernel_tile = tl.shared_tensor(tl.arange(0, BLOCK_SIZE), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Compute convolution for this output position
    for k in range(kernel_size):
        for l in range(kernel_size):
            # Calculate input coordinates
            input_row = output_row * stride + k - padding
            input_col = output_col * stride + l - padding
            
            # Check bounds
            if input_row >= 0 and input_row < input_height and input_col >= 0 and input_col < input_width:
                # Calculate input index
                input_idx = batch_idx * (in_channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           input_row * input_width + input_col
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load kernel value
                kernel_val = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + k * kernel_size + l, mask=True)
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store result
    tl.store(output_ptr + output_pos, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,
        in_channels,
        output_height,
        output_width
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
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
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
        # Use our Triton implementation
        output = triton_depthwise_conv2d(x, self.weight, self.stride, self.padding)
        
        # Add bias if present
        if self.bias is not None:
            # Expand bias to match output shape
            bias_expanded = self.bias.view(1, -1, 1, 1)
            output = output + bias_expanded
            
        return output