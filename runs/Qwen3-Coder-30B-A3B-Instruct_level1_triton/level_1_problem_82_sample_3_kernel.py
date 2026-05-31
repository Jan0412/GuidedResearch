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
    output_block_id = tl.program_id(2)
    
    # Calculate output indices
    output_elements_per_thread = OUTPUT_ELEMENTS_PER_BLOCK // BLOCK_SIZE
    thread_idx = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.float32, (CHANNELS_PER_BLOCK, kernel_size, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Calculate starting position in output
    start_output_idx = output_block_id * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Process each kernel element
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input positions
            input_h_start = tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) // output_width * stride - padding + k_h
            input_w_start = tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) % output_width * stride - padding + k_w
            
            # Bounds checking
            valid_mask = (input_h_start >= 0) & (input_h_start < input_height) & \
                        (input_w_start >= 0) & (input_w_start < input_width)
            
            # Load input values
            input_vals = tl.load(input_ptr + 
                               batch_id * (in_channels * input_height * input_width) +
                               channel_id * (input_height * input_width) +
                               input_h_start * input_width + input_w_start,
                               mask=valid_mask, other=0.0)
            
            # Load weight value
            weight_val = tl.load(weight_ptr + channel_id * kernel_size * kernel_size + k_h * kernel_size + k_w)
            
            # Accumulate
            acc += input_vals * weight_val
    
    # Store results
    output_start = batch_id * (in_channels * output_height * output_width) + \
                  channel_id * (output_height * output_width) + start_output_idx
    tl.store(output_ptr + output_start + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK),
             acc, mask=tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) < OUTPUT_ELEMENTS_PER_BLOCK)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Kernel configuration
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 64
    
    # Grid dimensions
    grid = (
        batch_size,
        in_channels,
        (output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK,
        (OUTPUT_ELEMENTS_PER_BLOCK + BLOCK_SIZE - 1) // BLOCK_SIZE
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
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
        
        # Initialize weights and bias
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
        # Use Triton kernel for depthwise convolution
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)