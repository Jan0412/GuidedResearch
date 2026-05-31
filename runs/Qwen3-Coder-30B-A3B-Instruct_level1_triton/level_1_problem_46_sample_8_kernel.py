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
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate total output elements
    total_output_elements = batch_size * channels * output_depth * output_height * output_width
    
    if idx >= total_output_elements:
        return
        
    # Convert linear index to multi-dimensional indices
    batch_idx = idx // (channels * output_depth * output_height * output_width)
    remaining = idx % (channels * output_depth * output_height * output_width)
    channel_idx = remaining // (output_depth * output_height * output_width)
    remaining = remaining % (output_depth * output_height * output_width)
    depth_idx = remaining // (output_height * output_width)
    remaining = remaining % (output_height * output_width)
    height_idx = remaining // output_width
    width_idx = remaining % output_width
    
    # Calculate input region boundaries
    input_depth_start = depth_idx * stride - padding
    input_height_start = height_idx * stride - padding
    input_width_start = width_idx * stride - padding
    
    # Initialize sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for k in range(kernel_size):
        for j in range(kernel_size):
            for i in range(kernel_size):
                input_d = input_depth_start + k
                input_h = input_height_start + j
                input_w = input_width_start + i
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (
                        batch_idx * (channels * input_depth * input_height * input_width) +
                        channel_idx * (input_depth * input_height * input_width) +
                        input_d * (input_height * input_width) +
                        input_h * input_width +
                        input_w
                    )
                    
                    sum_val += tl.load(input_ptr + input_idx, mask=True)
                    count += 1
    
    # Calculate average
    if count > 0:
        avg = sum_val / count
    else:
        avg = 0.0
    
    # Write output
    output_idx = idx
    tl.store(output_ptr + output_idx, avg)

def triton_avgpool3d(input_tensor, kernel_size, stride, padding):
    """
    Triton implementation of 3D Average Pooling
    """
    batch_size, channels, depth, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure output dimensions are valid
    if output_depth <= 0 or output_height <= 0 or output_width <= 0:
        raise ValueError("Invalid output dimensions")
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_depth, output_height, output_width, 
                         dtype=torch.float32, device=input_tensor.device)
    
    # Flatten tensors for kernel execution
    flat_input = input_tensor.contiguous().view(-1)
    flat_output = output.view(-1)
    
    # Configure kernel launch parameters
    total_elements = batch_size * channels * output_depth * output_height * output_width
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    avgpool3d_kernel[grid_size](
        flat_input,
        flat_output,
        batch_size,
        channels,
        depth,
        height,
        width,
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
    Optimized Model using Triton kernels for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

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
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)