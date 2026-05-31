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
    
    # Create masks for valid channels
    channel_mask = global_channel < in_channels
    
    # Create masks for valid spatial locations
    height_mask = (global_height >= 0) & (global_height < height_in)
    width_mask = (global_width >= 0) & (global_width < width_in)
    
    # Combine all masks
    valid_mask = channel_mask & height_mask & width_mask
    
    # Load input
    input_offset = batch_idx * (in_channels * height_in * width_in) + \
                   global_channel * (height_in * width_in) + \
                   global_height * width_in + global_width
    
    input_vals = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((CHANNELS_PER_BLOCK, HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Convolution loop
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input positions
            h_offset = k_h - padding
            w_offset = k_w - padding
            
            # Calculate input indices with stride
            input_h = global_height * stride + h_offset
            input_w = global_width * stride + w_offset
            
            # Create input mask
            input_h_valid = (input_h >= 0) & (input_h < height_in)
            input_w_valid = (input_w >= 0) & (input_w < width_in)
            input_valid = input_h_valid & input_w_valid
            
            # Load weight
            weight_val = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + k_h * kernel_size + k_w)
            
            # Calculate input offset
            input_offset = batch_idx * (in_channels * height_in * width_in) + \
                           global_channel * (height_in * width_in) + \
                           input_h * width_in + input_w
            
            # Load input with appropriate masking
            input_val = tl.load(input_ptr + input_offset, mask=input_valid & channel_mask, other=0.0)
            
            # Perform convolution operation
            acc += weight_val * input_val
    
    # Store output
    output_offset = batch_idx * (in_channels * height_out * width_out) + \
                    global_channel * (height_out * width_out) + \
                    global_height * width_out + global_width
    
    # Create final output mask
    output_mask = channel_mask & (global_height < height_out) & (global_width < width_out)
    
    tl.store(output_ptr + output_offset, acc, mask=output_mask)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, height_out, width_out, dtype=torch.float32, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 16
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (height_out + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (width_out + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    Optimized with Triton kernels for improved performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None}'