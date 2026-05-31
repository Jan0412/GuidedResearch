import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avg_pool3d_kernel(
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
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_d = tl.program_id(2)
    
    # Calculate global output indices
    output_h = tl.program_id(3)
    output_w = tl.program_id(4)
    
    # Ensure we're within bounds
    if output_d >= output_depth or output_h >= output_height or output_w >= output_width:
        return
    
    # Calculate input start positions with padding
    input_start_d = output_d * stride - padding
    input_start_h = output_h * stride - padding
    input_start_w = output_w * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position
                input_d = input_start_d + kd
                input_h = input_start_h + kh
                input_w = input_start_w + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (batch_idx * (channels * input_depth * input_height * input_width) +
                                channel_idx * (input_depth * input_height * input_width) +
                                input_d * (input_height * input_width) +
                                input_h * input_width +
                                input_w)
                    
                    # Load value and accumulate
                    val = tl.load(input_ptr + input_idx, mask=True)
                    acc += val
                    count += 1
    
    # Calculate output index
    output_idx = (batch_idx * (channels * output_depth * output_height * output_width) +
                  channel_idx * (output_depth * output_height * output_width) +
                  output_d * (output_height * output_width) +
                  output_h * output_width +
                  output_w)
    
    # Store result
    if count > 0:
        avg_val = acc / count
        tl.store(output_ptr + output_idx, avg_val)
    else:
        tl.store(output_ptr + output_idx, 0.0)

def triton_avg_pool3d(input_tensor, kernel_size, stride, padding):
    """
    Custom Triton implementation of 3D average pooling.
    """
    # Ensure input is on CUDA
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA")
    
    # Get dimensions
    batch_size, channels, depth, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, channels, output_depth, output_height, output_width, 
                        dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel
    grid = (
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width
    )
    
    BLOCK_SIZE = 1024
    avg_pool3d_kernel[grid](
        input_tensor,
        output,
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
    Optimized model using custom Triton kernels for 3D Average Pooling.
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
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)