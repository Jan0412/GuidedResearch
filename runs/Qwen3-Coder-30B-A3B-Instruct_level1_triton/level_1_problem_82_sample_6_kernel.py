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
    output_elements_per_block = OUTPUT_ELEMENTS_PER_BLOCK
    output_start_idx = output_block_id * output_elements_per_block
    
    # Shared memory for input tile
    input_tile = tl.shared.tensor([CHANNELS_PER_BLOCK, kernel_size, kernel_size], tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Process each kernel element
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input position
            input_y_start = (output_start_idx // output_width) * stride - padding + k_h
            input_x_start = (output_start_idx % output_width) * stride - padding + k_w
            
            # Check bounds
            valid_y = (input_y_start >= 0) & (input_y_start < input_height)
            valid_x = (input_x_start >= 0) & (input_x_start < input_width)
            
            # Load input data
            if valid_y & valid_x:
                input_val = tl.load(input_ptr + 
                    batch_id * (in_channels * input_height * input_width) +
                    channel_id * (input_height * input_width) +
                    input_y_start * input_width + input_x_start)
            else:
                input_val = 0.0
                
            # Load weight
            weight_val = tl.load(weight_ptr + 
                channel_id * (kernel_size * kernel_size) +
                k_h * kernel_size + k_w)
                
            # Accumulate
            acc += input_val * weight_val
    
    # Store results
    output_indices = output_start_idx + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
    valid_mask = output_indices < (output_height * output_width)
    
    tl.store(output_ptr + 
        batch_id * (in_channels * output_height * output_width) +
        channel_id * (output_height * output_width) +
        output_indices,
        acc,
        mask=valid_mask)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 16
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
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
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Optimized using Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and biases
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
        # Use Triton kernel instead of PyTorch's native conv2d
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

# For compatibility with the test harness
def get_inputs():
    batch_size = 16
    in_channels = 64
    kernel_size = 3
    width = 512
    height = 512
    stride = 1
    padding = 0
    
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 3, 1, 0]