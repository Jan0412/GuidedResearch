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
):
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate output indices
    output_h = tl.program_id(2)
    output_w = tl.program_id(3)
    
    # Check bounds
    if output_h >= output_height or output_w >= output_width:
        return
    
    # Calculate input start positions with padding
    input_start_h = output_h * stride_h - padding_h
    input_start_w = output_w * stride_w - padding_w
    
    # Initialize accumulator
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input position
            ih = input_start_h + kh
            iw = input_start_w + kw
            
            # Check if within bounds (including padding)
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Calculate input index
                input_idx = batch_idx * (channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           ih * input_width + iw
                
                # Load value and accumulate
                val = tl.load(input_ptr + input_idx, mask=True)
                sum_val += val
                count += 1
    
    # Calculate average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Calculate output index
    output_idx = batch_idx * (channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_h * output_width + output_w
    
    # Store result
    tl.store(output_ptr + output_idx, avg_val)

def triton_avg_pool2d(input_tensor, kernel_size, stride=None, padding=0):
    """
    Triton implementation of 2D Average Pooling
    """
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
        
    if stride is None:
        stride_h = stride_w = kernel_h
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Ensure output dimensions are valid
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Invalid pooling parameters")
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (
        batch_size,
        channels,
        output_height,
        output_width
    )
    
    # Define block size
    BLOCK_SIZE = 128
    
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
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with Triton optimization.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)