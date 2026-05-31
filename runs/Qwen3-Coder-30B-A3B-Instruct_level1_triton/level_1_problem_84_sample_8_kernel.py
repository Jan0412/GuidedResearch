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
    
    # Calculate global positions
    global_h = height_idx * HEIGHT_PER_BLOCK + tl.arange(0, HEIGHT_PER_BLOCK)[:, None]
    global_w = width_idx * WIDTH_PER_BLOCK + tl.arange(0, WIDTH_PER_BLOCK)[None, :]
    
    # Bounds checking
    h_mask = (global_h < height_out) & (global_h >= 0)
    w_mask = (global_w < width_out) & (global_w >= 0)
    hw_mask = h_mask & w_mask
    
    # Shared memory for input tile
    input_tile = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Process multiple channels if needed
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Check if this channel block is valid
        channel_mask = (channel_idx * CHANNELS_PER_BLOCK + tl.arange(0, CHANNELS_PER_BLOCK)) < in_channels
        
        # Load weight for current channel
        weight = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + 
                        tl.arange(0, kernel_size)[:, None] * kernel_size + 
                        tl.arange(0, kernel_size)[None, :])
        
        # Load input data with padding
        input_h_start = global_h * stride - padding
        input_w_start = global_w * stride - padding
        
        # Gather input values with proper boundary handling
        input_values = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
        
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = input_h_start + kh
                iw = input_w_start + kw
                
                # Boundary check for input
                input_h_valid = (ih >= 0) & (ih < height_in)
                input_w_valid = (iw >= 0) & (iw < width_in)
                valid = input_h_valid & input_w_valid
                
                # Load input value
                input_val = tl.load(input_ptr + batch_idx * in_channels * height_in * width_in +
                                  (c + channel_idx * CHANNELS_PER_BLOCK) * height_in * width_in +
                                  ih * width_in + iw, mask=valid & (hw_mask if not (kh == 0 and kw == 0) else True), other=0.0)
                
                # Accumulate convolution
                input_values += input_val * weight[kh, kw]
        
        # Accumulate across channels
        if c == 0:
            output = input_values
        else:
            output += input_values
    
    # Store output
    if channel_idx < in_channels:
        output = tl.load(output_ptr + batch_idx * in_channels * height_out * width_out +
                        channel_idx * height_out * width_out + 
                        global_h * width_out + global_w, mask=hw_mask, other=0.0)
        tl.store(output_ptr + batch_idx * in_channels * height_out * width_out +
                channel_idx * height_out * width_out + 
                global_h * width_out + global_w, output, mask=hw_mask)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_height) // stride + 1
    width_out = (width_in + 2 * padding - kernel_width) // stride + 1
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
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
        kernel_height,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)