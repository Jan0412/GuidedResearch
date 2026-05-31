import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avgpool2d_kernel(
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
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate output dimensions
    total_elements = batch_size * channels * output_height * output_width
    
    # Each thread block processes one output element
    output_x = tl.program_id(3)
    
    # Global output index
    global_output_idx = batch_idx * (channels * output_height * output_width) + \
                        channel_idx * (output_height * output_width) + \
                        output_y * output_width + output_x
    
    if global_output_idx >= total_elements:
        return
    
    # Calculate input region boundaries
    input_y_start = output_y * stride_h - padding_h
    input_x_start = output_x * stride_w - padding_w
    
    # Initialize sum and count
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over the pooling window
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            input_y = input_y_start + ky
            input_x = input_x_start + kx
            
            # Check if within input bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Calculate input index
                input_idx = batch_idx * (channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           input_y * input_width + input_x
                
                # Load value and accumulate
                val = tl.load(input_ptr + input_idx, mask=True)
                sum_val += val
                count += 1
    
    # Compute average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    tl.store(output_ptr + global_output_idx, avg_val)

def triton_avgpool2d(input_tensor, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w):
    """
    Custom Triton implementation of 2D Average Pooling
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Define grid configuration
    grid = (
        batch_size,           # batch dimension
        channels,             # channel dimension  
        output_height,        # output height dimension
        output_width          # output width dimension
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 1
    
    avgpool2d_kernel[grid](
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for 2D Average Pooling
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with custom Triton kernel.

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
        Applies 2D Average Pooling using custom Triton kernel.

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