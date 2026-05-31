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
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    
    # Calculate global indices
    global_channel = channel_idx * CHANNELS_PER_BLOCK + tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None]
    global_height = height_idx * HEIGHT_PER_BLOCK + tl.arange(0, HEIGHT_PER_BLOCK)[None, :, None]
    global_width = width_idx * WIDTH_PER_BLOCK + tl.arange(0, WIDTH_PER_BLOCK)[None, None, :]
    
    # Mask for valid channels
    channel_mask = global_channel < in_channels
    
    # Calculate output positions
    out_h_start = global_height * stride - padding
    out_w_start = global_width * stride - padding
    
    # Check if we're within bounds
    valid_h = (out_h_start >= 0) & (out_h_start + kernel_size <= height_in)
    valid_w = (out_w_start >= 0) & (out_w_start + kernel_size <= width_in)
    valid = channel_mask & valid_h & valid_w
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Perform convolution
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input positions
            in_h = out_h_start + k_h
            in_w = out_w_start + k_w
            
            # Load input data
            input_data = tl.load(input_ptr + 
                               batch_idx * (in_channels * height_in * width_in) +
                               global_channel * (height_in * width_in) +
                               in_h * width_in +
                               in_w,
                               mask=valid, other=0.0)
            
            # Load weight data
            weight_data = tl.load(weight_ptr + 
                                global_channel * kernel_size * kernel_size +
                                k_h * kernel_size +
                                k_w,
                                mask=channel_mask, other=0.0)
            
            # Accumulate
            acc += input_data * weight_data[:, None, None]
    
    # Store output
    output_ptr_idx = batch_idx * (in_channels * height_out * width_out) + \
                    global_channel * (height_out * width_out) + \
                    global_height * width_out + \
                    global_width
    
    tl.store(output_ptr + output_ptr_idx, acc, mask=valid)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Pad input if necessary
    if padding > 0:
        input_padded = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))
    else:
        input_padded = input_tensor
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 16
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (height_out + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (width_out + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_padded,
        weight,
        output,
        batch_size,
        in_channels,
        height_in + 2 * padding,
        width_in + 2 * padding,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernels for depthwise convolution
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel
        """
        # Convert to float32 for consistency
        if x.dtype != torch.float32:
            x = x.float()
        
        # Call our Triton implementation
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

# Keep original class for comparison
class Model(nn.Module):
    """
    Original implementation with PyTorch Conv2d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)