import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    input_length,  # Input length
    output_length,  # Output length
    kernel_size,  # Pooling kernel size
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # We'll process one (batch, channel) pair per program
    # Total number of (batch, channel) pairs
    n_batches = batch_size * in_channels
    
    # Program ID corresponds to which (batch, channel) pair we're processing
    batch_id = tl.program_id(0)
    
    # Skip if beyond bounds
    if batch_id >= n_batches:
        return
        
    # Compute actual batch and channel indices
    b = batch_id // in_channels
    c = batch_id % in_channels
    
    # Calculate input and output pointers for this (batch, channel)
    # Input: [batch, channel, :] starts at x_ptr + (b * in_channels + c) * input_length
    x_offset = (b * in_channels + c) * input_length
    # Output: [batch, channel, :] starts at out_ptr + (b * in_channels + c) * output_length
    out_offset = (b * in_channels + c) * output_length
    
    x_ptr_b = x_ptr + x_offset
    out_ptr_c = out_ptr + out_offset
    
    # Iterate over output positions
    for out_idx in range(output_length):
        # Calculate the start and end indices in the input for this output position
        # Taking padding into account: actual start in input = out_idx * stride - padding
        start = out_idx * stride - padding
        end = start + kernel_size
        
        # Clamp to valid input range
        start_clamped = tl.maximum(0, start)
        end_clamped = tl.minimum(input_length, end)
        
        # Compute effective kernel size (accounting for boundaries at edges)
        effective_kernel_size = end_clamped - start_clamped
        
        # Compute sum over the valid range
        sum_val = 0.0
        for i in range(start_clamped, end_clamped):
            sum_val += tl.load(x_ptr_b + i)
        
        # Compute average
        avg_val = sum_val / effective_kernel_size
        
        # Store result
        tl.store(out_ptr_c + out_idx, avg_val)


def triton_avg_pool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    """
    Triton implementation of 1D average pooling.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size (int): Size of the pooling window
        stride (int): Stride of the pooling operation
        padding (int): Padding applied to the input
        
    Returns:
        torch.Tensor: Output tensor with 1D Average Pooling applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length: floor((input_length + 2*padding - kernel_size) / stride) + 1
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    # Grid: one program per (batch, channel) pair
    n_programs = batch_size * in_channels
    BLOCK_SIZE = 128  # Not used in this kernel, but kept for consistency
    
    # Launch the Triton kernel
    avg_pool1d_kernel[n_programs](x, out, batch_size, in_channels, input_length, 
                                  output_length, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model using a custom Triton kernel for 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the optimized 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized 1D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)