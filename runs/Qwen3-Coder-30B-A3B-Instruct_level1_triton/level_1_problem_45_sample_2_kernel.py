import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avgpool2d_kernel(
    input_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get the batch and channel index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Each thread handles one output element
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate output dimensions
    total_output_elements = output_height * output_width
    
    # Only process valid output indices
    mask = output_idx < total_output_elements
    
    # Convert linear index to 2D coordinates
    out_y = output_idx // output_width
    out_x = output_idx % output_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process all kernel elements for this output position
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates
            in_y = out_y * stride_h - padding_h + kh
            in_x = out_x * stride_w - padding_w + kw
            
            # Check if input coordinate is within bounds
            in_bounds = (in_y >= 0) & (in_y < input_height) & (in_x >= 0) & (in_x < input_width)
            
            # Load input value if valid
            input_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            if in_bounds:
                input_offset = batch_idx * (channels * input_height * input_width) + \
                              channel_idx * (input_height * input_width) + \
                              in_y * input_width + in_x
                input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            
            acc += input_val
    
    # Divide by kernel area to compute average
    kernel_area = kernel_h * kernel_w
    avg_val = acc / kernel_area
    
    # Store output
    output_offset = batch_idx * (channels * output_height * output_width) + \
                   channel_idx * (output_height * output_width) + output_idx
    tl.store(output_ptr + output_offset, avg_val, mask=mask)

def triton_avgpool2d(input_tensor, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w):
    """
    Triton implementation of 2D Average Pooling
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Ensure output dimensions are positive
    assert output_height > 0 and output_width > 0, "Invalid pooling parameters"
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 1
    
    # Grid configuration
    grid_batch = batch_size
    grid_channels = channels
    grid_output = (output_height * output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    avgpool2d_kernel[
        (grid_batch, grid_channels, grid_output),
        num_warps=4,
        num_stages=3
    ](
        input_tensor,
        output,
        height,
        width,
        output_height,
        output_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        batch_size,
        channels,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with optimized Triton implementation.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool2d(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )