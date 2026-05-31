import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output width index
    output_w_idx = tl.program_id(3)
    
    # Ensure we're within bounds
    if output_h_idx >= output_height or output_w_idx >= output_width:
        return
    
    # Shared memory for reduction
    shared_data = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Calculate input coordinates
    start_h = output_h_idx * stride_h - padding_h
    start_w = output_w_idx * stride_w - padding_w
    
    # Initialize sum
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = start_h + kh
            iw = start_w + kw
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Calculate input index
                input_idx = batch_idx * (channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           ih * input_width + iw
                
                # Load value
                val = tl.load(input_ptr + input_idx, mask=True)
                sum_val += val
                count += 1
    
    # Normalize by count
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Write output
    output_idx = batch_idx * (channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_h_idx * output_width + output_w_idx
    
    tl.store(output_ptr + output_idx, avg_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

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
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, dtype=torch.float32, device=x.device)
        
        # Prepare input tensor for contiguous access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Define grid dimensions
        grid = (
            batch_size,
            channels,
            output_height,
            output_width
        )
        
        # Launch kernel
        BLOCK_SIZE = 256
        CHANNELS_PER_BLOCK = 4
        
        avg_pool2d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            height,
            width,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
        
        return output

# For the specific case in the problem
batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [kernel_size]