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
    output_idx = tl.program_id(2)
    
    # Calculate output coordinates
    output_d = output_idx // (output_height * output_width)
    remaining = output_idx % (output_height * output_width)
    output_h = remaining // output_width
    output_w = remaining % output_width
    
    # Calculate input start positions
    input_d_start = output_d * stride - padding
    input_h_start = output_h * stride - padding
    input_w_start = output_w * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate through kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_d = input_d_start + kd
                input_h = input_h_start + kh
                input_w = input_w_start + kw
                
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
                    
                    acc += tl.load(input_ptr + input_idx, mask=True)
                    count += 1
    
    # Calculate average
    if count > 0:
        avg_val = acc / count
    else:
        avg_val = 0.0
    
    # Calculate output index
    output_idx_total = (batch_idx * (channels * output_depth * output_height * output_width) +
                       channel_idx * (output_depth * output_height * output_width) +
                       output_d * (output_height * output_width) +
                       output_h * output_width +
                       output_w)
    
    tl.store(output_ptr + output_idx_total, avg_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 3D Average Pooling.
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
        batch_size, channels, depth, height, width = x.shape
        
        # Calculate output dimensions
        output_depth = (depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_depth, output_height, output_width, 
                           dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Launch kernel
        grid = (
            batch_size,
            channels,
            output_depth * output_height * output_width
        )
        
        BLOCK_SIZE = 128
        avg_pool3d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            depth,
            height,
            width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size,
            self.stride,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

# Helper function to compute output dimensions
def compute_output_dims(input_shape, kernel_size, stride, padding):
    batch_size, channels, depth, height, width = input_shape
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    return (batch_size, channels, output_depth, output_height, output_width)