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
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_element_id = tl.program_id(2)
    
    # Calculate global output indices
    output_h = output_element_id // output_width
    output_w = output_element_id % output_width
    
    # Check bounds
    if output_h >= output_height or output_w >= output_width:
        return
        
    # Calculate input region start positions
    input_h_start = output_h * stride - padding
    input_w_start = output_w * stride - padding
    
    # Shared memory for input tile and kernel
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    shared_kernel = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel spatial dimensions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input coordinates
            ih = input_h_start + kh
            iw = input_w_start + kw
            
            # Check if input coordinate is valid
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input value
                input_idx = batch_id * (in_channels * input_height * input_width) + \
                           channel_id * (input_height * input_width) + \
                           ih * input_width + iw
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load kernel value
                kernel_idx = channel_id * kernel_size * kernel_size + kh * kernel_size + kw
                kernel_val = tl.load(weight_ptr + kernel_idx, mask=True)
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store result
    output_idx = batch_id * (in_channels * output_height * output_width) + \
                channel_id * (output_height * output_width) + \
                output_h * output_width + output_w
    tl.store(output_ptr + output_idx, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        output_height * output_width
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
        BLOCK_SIZE=16,
        CHANNELS_PER_BLOCK=1,
        OUTPUT_ELEMENTS_PER_BLOCK=1
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
            self.bias = nn.Parameter(torch.randn(in_channels))
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
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(x, self.weight, self.stride, self.padding)
        
        # Add bias if applicable
        if self.bias is not None:
            # Expand bias to match output shape (batch, channels, height, width)
            bias_expanded = self.bias.view(1, -1, 1, 1)
            output = output + bias_expanded
            
        return output