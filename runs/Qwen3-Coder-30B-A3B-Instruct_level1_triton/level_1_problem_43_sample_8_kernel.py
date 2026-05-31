import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_d1,
    input_d2,
    input_d3,
    output_d1,
    output_d2,
    output_d3,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output coordinates
    output_d1_idx = output_idx // (output_d2 * output_d3)
    remaining = output_idx % (output_d2 * output_d3)
    output_d2_idx = remaining // output_d3
    output_d3_idx = remaining % output_d3
    
    # Calculate input region boundaries
    start_d1 = output_d1_idx * stride - padding
    start_d2 = output_d2_idx * stride - padding
    start_d3 = output_d3_idx * stride - padding
    
    # Apply dilation
    input_start_d1 = start_d1
    input_start_d2 = start_d2
    input_start_d3 = start_d3
    
    # Initialize maximum value
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    
    # Iterate through kernel
    for kd1 in range(kernel_size):
        for kd2 in range(kernel_size):
            for kd3 in range(kernel_size):
                # Calculate input positions with dilation
                input_d1_pos = input_start_d1 + kd1 * dilation
                input_d2_pos = input_start_d2 + kd2 * dilation
                input_d3_pos = input_start_d3 + kd3 * dilation
                
                # Check bounds
                if (input_d1_pos >= 0 and input_d1_pos < input_d1 and 
                    input_d2_pos >= 0 and input_d2_pos < input_d2 and 
                    input_d3_pos >= 0 and input_d3_pos < input_d3):
                    
                    # Calculate global index
                    input_idx = (
                        batch_idx * (channels * input_d1 * input_d2 * input_d3) +
                        channel_idx * (input_d1 * input_d2 * input_d3) +
                        input_d1_pos * (input_d2 * input_d3) +
                        input_d2_pos * input_d3 +
                        input_d3_pos
                    )
                    
                    # Load input value
                    val = tl.load(input_ptr + input_idx, mask=True)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    output_idx_global = (
        batch_idx * (channels * output_d1 * output_d2 * output_d3) +
        channel_idx * (output_d1 * output_d2 * output_d3) +
        output_d1_idx * (output_d2 * output_d3) +
        output_d2_idx * output_d3 +
        output_d3_idx
    )
    
    tl.store(output_ptr + output_idx_global, max_val)

def triton_maxpool3d(input_tensor, kernel_size, stride, padding, dilation):
    """
    Custom Triton implementation of 3D Max Pooling
    """
    batch_size, channels, input_d1, input_d2, input_d3 = input_tensor.shape
    
    # Calculate output dimensions
    output_d1 = (input_d1 + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    output_d2 = (input_d2 + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    output_d3 = (input_d3 + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Grid configuration
    num_blocks = batch_size * channels * output_d1 * output_d2 * output_d3
    grid = (batch_size, channels, num_blocks)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        input_d1,
        input_d2,
        input_d3,
        output_d1,
        output_d2,
        output_d3,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=1024
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version of Max Pooling 3D using Triton kernels
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # Use Triton kernel for max pooling
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)