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
    
    # Ensure we don't go out of bounds
    if output_h_idx >= output_height or output_w_idx >= output_width:
        return
    
    # Shared memory for reduction
    shared_mem = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate start positions in input tensor
    h_start = output_h_idx * stride_h - padding_h
    w_start = output_w_idx * stride_w - padding_w
    
    # Calculate effective kernel size
    kh_start = tl.max(tl.zeros((1,), dtype=tl.int32), -h_start)
    kw_start = tl.max(tl.zeros((1,), dtype=tl.int32), -w_start)
    kh_end = tl.min(kernel_h, input_height - h_start)
    kw_end = tl.min(kernel_w, input_width - w_start)
    
    # Count valid elements
    count = 0
    
    # Iterate through kernel
    for kh in range(kh_start, kh_end):
        for kw in range(kw_start, kw_end):
            ih = h_start + kh
            iw = w_start + kw
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Calculate global index
                idx = batch_idx * (channels * input_height * input_width) + \
                      channel_idx * (input_height * input_width) + \
                      ih * input_width + iw
                
                acc += tl.load(input_ptr + idx, mask=True)
                count += 1
    
    # Normalize by count
    if count > 0:
        acc = acc / count
    else:
        acc = 0.0
    
    # Store result
    output_idx = batch_idx * (channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_h_idx * output_width + output_w_idx
    
    tl.store(output_ptr + output_idx, acc)

def triton_avg_pool2d(input_tensor, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w):
    """
    Custom Triton implementation of 2D average pooling
    """
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        input_tensor,
        output,
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
        BLOCK_SIZE=1024,
        CHANNELS_PER_BLOCK=1
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for 2D Average Pooling.
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
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )

# For compatibility with existing interface
def get_inputs():
    batch_size = 16
    channels = 64
    height = 2048
    width = 2048
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    kernel_size = 11
    return [kernel_size]