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
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    height_id = tl.program_id(2)
    width_id = tl.program_id(3)
    
    # Calculate global indices
    global_h = height_id * HEIGHT_PER_BLOCK + tl.arange(0, HEIGHT_PER_BLOCK)[:, None]
    global_w = width_id * WIDTH_PER_BLOCK + tl.arange(0, WIDTH_PER_BLOCK)[None, :]
    
    # Bounds checking for output dimensions
    h_mask = (global_h < height_out) & (global_h >= 0)
    w_mask = (global_w < width_out) & (global_w >= 0)
    hw_mask = h_mask & w_mask
    
    # Shared memory for input tile and kernel
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding, WIDTH_PER_BLOCK + 2 * padding))
    shared_kernel = tl.shared_memory(dtype=tl.float32, shape=(kernel_size, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel spatial dimensions
    for kh in range(0, kernel_size):
        for kw in range(0, kernel_size):
            # Calculate input coordinates
            ih = global_h * stride + kh - padding
            iw = global_w * stride + kw - padding
            
            # Load kernel weights
            k_val = tl.load(weight_ptr + channel_id * kernel_size * kernel_size + kh * kernel_size + kw, mask=True)
            
            # Load input data with boundary checks
            input_mask = (ih >= 0) & (ih < height_in) & (iw >= 0) & (iw < width_in)
            input_val = tl.load(input_ptr + batch_id * in_channels * height_in * width_in +
                               channel_id * height_in * width_in + 
                               ih * width_in + iw, mask=input_mask, other=0.0)
            
            # Accumulate
            acc += input_val * k_val
    
    # Store output
    if hw_mask.any():
        output_idx = batch_id * in_channels * height_out * width_out + \
                     channel_id * height_out * width_out + \
                     global_h * width_out + global_w
        tl.store(output_ptr + output_idx, acc, mask=hw_mask)

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    kernel_size = weight.shape[-1]  # Assuming square kernel
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 16
    WIDTH_PER_BLOCK = 16
    
    # Grid dimensions
    grid = (
        batch_size,           # Batch dimension
        in_channels,          # Channel dimension
        (height_out + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,  # Height blocks
        (width_out + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK     # Width blocks
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
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights
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
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(x, self.weight, self.stride, self.padding)
        
        # Add bias if applicable
        if self.bias is not None:
            # Expand bias to match output shape
            bias_expanded = self.bias.view(1, -1, 1, 1)
            output = output + bias_expanded
            
        return output