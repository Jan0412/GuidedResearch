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
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    
    # Calculate global indices
    global_channel = channel_idx * CHANNELS_PER_BLOCK + tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None]
    global_height = height_idx * HEIGHT_PER_BLOCK + tl.arange(0, HEIGHT_PER_BLOCK)[None, :, None]
    global_width = width_idx * WIDTH_PER_BLOCK + tl.arange(0, WIDTH_PER_BLOCK)[None, None, :]
    
    # Create masks for valid regions
    channel_mask = global_channel < in_channels
    height_mask = global_height < height_out
    width_mask = global_width < width_out
    
    # Combine all masks
    valid_mask = channel_mask & height_mask & width_mask
    
    # Initialize accumulator
    acc = tl.zeros((CHANNELS_PER_BLOCK, HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Perform convolution
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input positions
            input_h = global_height * stride - padding + k_h
            input_w = global_width * stride - padding + k_w
            
            # Create input masks
            input_h_valid = (input_h >= 0) & (input_h < height_in)
            input_w_valid = (input_w >= 0) & (input_w < width_in)
            input_valid_mask = input_h_valid & input_w_valid
            
            # Load input data
            input_data = tl.load(
                input_ptr + 
                batch_idx * (in_channels * height_in * width_in) +
                global_channel * (height_in * width_in) +
                input_h * width_in +
                input_w,
                mask=input_valid_mask & channel_mask,
                other=0.0
            )
            
            # Load weight data
            weight_data = tl.load(
                weight_ptr + 
                global_channel * kernel_size * kernel_size +
                k_h * kernel_size +
                k_w,
                mask=channel_mask,
                other=0.0
            )
            
            # Multiply and accumulate
            acc += input_data * weight_data[:, None, None]
    
    # Store result
    tl.store(
        output_ptr + 
        batch_idx * (in_channels * height_out * width_out) +
        global_channel * (height_out * width_out) +
        global_height * width_out +
        global_width,
        acc,
        mask=valid_mask
    )

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    out_channels = weight.shape[0]
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 256
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
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    Optimized with Triton kernels for better performance.
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