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
    out_h_start = height_idx * HEIGHT_PER_BLOCK
    out_w_start = width_idx * WIDTH_PER_BLOCK
    
    # Shared memory for input tile and weight
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding, WIDTH_PER_BLOCK + 2 * padding))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(kernel_size, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel size
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input position
            h_start = out_h_start * stride - padding + k_h
            w_start = out_w_start * stride - padding + k_w
            
            # Load weight
            weight_val = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + k_h * kernel_size + k_w)
            
            # Load input region (with boundary checks)
            for h in range(HEIGHT_PER_BLOCK):
                for w in range(WIDTH_PER_BLOCK):
                    if h_start + h >= 0 and h_start + h < height_in and w_start + w >= 0 and w_start + w < width_in:
                        input_val = tl.load(input_ptr + 
                                          batch_idx * (in_channels * height_in * width_in) +
                                          channel_idx * (height_in * width_in) +
                                          (h_start + h) * width_in +
                                          (w_start + w))
                    else:
                        input_val = 0.0
                    
                    acc[h, w] += input_val * weight_val
    
    # Write output
    for h in range(HEIGHT_PER_BLOCK):
        for w in range(WIDTH_PER_BLOCK):
            if out_h_start + h < height_out and out_w_start + w < width_out:
                tl.store(output_ptr + 
                        batch_idx * (in_channels * height_out * width_out) +
                        channel_idx * (height_out * width_out) +
                        (out_h_start + h) * width_out +
                        (out_w_start + w),
                        acc[h, w])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    if bias is not None:
        bias = bias.contiguous().cuda()
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, dtype=torch.float32, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 16
    WIDTH_PER_BLOCK = 16
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        in_channels,
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
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)