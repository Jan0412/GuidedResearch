import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avgpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate global output index
    output_depth_idx = output_idx // (output_height * output_width)
    remaining = output_idx % (output_height * output_width)
    output_height_idx = remaining // output_width
    output_width_idx = remaining % output_width
    
    # Calculate input region boundaries
    input_start_d = output_depth_idx * stride - padding
    input_start_h = output_height_idx * stride - padding
    input_start_w = output_width_idx * stride - padding
    
    # Initialize sum accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_d = input_start_d + kd
                input_h = input_start_h + kh
                input_w = input_start_w + kw
                
                # Check if within bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input linear index
                    input_linear_idx = (
                        batch_idx * (channels * input_depth * input_height * input_width) +
                        channel_idx * (input_depth * input_height * input_width) +
                        input_d * (input_height * input_width) +
                        input_h * input_width +
                        input_w
                    )
                    
                    # Load value and accumulate
                    val = tl.load(input_ptr + input_linear_idx, mask=True)
                    sum_val += val
                    count += 1
    
    # Calculate average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Calculate output linear index
    output_linear_idx = (
        batch_idx * (channels * output_depth * output_height * output_width) +
        channel_idx * (output_depth * output_height * output_width) +
        output_depth_idx * (output_height * output_width) +
        output_height_idx * output_width +
        output_width_idx
    )
    
    # Store result
    tl.store(output_ptr + output_linear_idx, avg_val)

def triton_avgpool3d(input_tensor, kernel_size, stride, padding):
    """
    Custom Triton implementation of 3D Average Pooling
    """
    batch_size, channels, input_depth, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding - kernel_size) // stride + 1
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_depth, output_height, output_width, 
                         dtype=torch.float32, device=input_tensor.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    num_threads = batch_size * channels * output_depth * output_height * output_width
    
    # Grid dimensions
    grid = (
        batch_size,           # batch dimension
        channels,             # channel dimension  
        output_depth * output_height * output_width  # output spatial dimensions
    )
    
    # Launch kernel
    avgpool3d_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with custom Triton implementation.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)