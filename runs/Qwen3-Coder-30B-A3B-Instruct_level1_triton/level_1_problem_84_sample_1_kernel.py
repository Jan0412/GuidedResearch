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
    
    # Bounds checking for channels
    channel_mask = global_channel < in_channels
    
    # Bounds checking for output dimensions
    height_mask = (global_height >= 0) & (global_height < height_out)
    width_mask = (global_width >= 0) & (global_width < width_out)
    valid_mask = channel_mask & height_mask & width_mask
    
    # Load input patch
    input_patches = tl.zeros((CHANNELS_PER_BLOCK, HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK))
    
    # Compute input coordinates
    input_height_start = global_height * stride - padding
    input_width_start = global_width * stride - padding
    
    # Iterate over kernel size
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = input_height_start + kh
            iw = input_width_start + kw
            
            # Check if input position is valid
            input_valid = (ih >= 0) & (ih < height_in) & (iw >= 0) & (iw < width_in)
            
            # Load input data
            input_data = tl.load(
                input_ptr + 
                batch_idx * (in_channels * height_in * width_in) +
                global_channel * (height_in * width_in) +
                ih * width_in +
                iw,
                mask=input_valid & channel_mask,
                other=0.0
            )
            
            # Load weight
            weight_data = tl.load(
                weight_ptr + 
                global_channel * (kernel_size * kernel_size) +
                kh * kernel_size +
                kw,
                mask=channel_mask,
                other=0.0
            )
            
            # Accumulate convolution
            input_patches += input_data * weight_data
    
    # Store output
    output_ptr = output_ptr + (
        batch_idx * (in_channels * height_out * width_out) +
        global_channel * (height_out * width_out) +
        global_height * width_out +
        global_width
    )
    
    tl.store(output_ptr, input_patches, mask=valid_mask)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, dtype=torch.float32, device='cuda')
    
    # Define block sizes
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
    Optimized with custom Triton kernels.
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