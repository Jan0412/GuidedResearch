import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this thread
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Only process valid output indices
    mask = output_idx < output_length
    
    # Calculate the starting input positions for each output element
    input_start_positions = output_idx * stride - padding
    
    # Initialize accumulator
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # For each kernel position, accumulate values
    for k in range(kernel_size):
        input_pos = input_start_positions + k
        
        # Check if input position is valid (within padded input bounds)
        input_valid_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Combine both masks
        combined_mask = mask & input_valid_mask
        
        # Calculate the actual input index
        input_idx = batch_idx * (in_channels * input_length) + channel_idx * input_length + input_pos
        
        # Load input value if valid
        input_val = tl.load(input_ptr + input_idx, mask=combined_mask, other=0.0)
        
        # Accumulate
        sum_val += input_val
    
    # Calculate the effective kernel size (number of valid elements)
    effective_kernel_size = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for k in range(kernel_size):
        input_pos = input_start_positions + k
        input_valid_mask = (input_pos >= 0) & (input_pos < input_length)
        effective_kernel_size += tl.where(input_valid_mask, 1, 0)
    
    # Avoid division by zero
    effective_kernel_size = tl.where(effective_kernel_size == 0, 1, effective_kernel_size)
    
    # Compute average
    avg_val = sum_val / effective_kernel_size.to(tl.float32)
    
    # Store result
    output_idx_global = batch_idx * (in_channels * output_length) + channel_idx * output_length + output_idx
    tl.store(output_ptr + output_idx_global, avg_val, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0):
    """
    Triton implementation of 1D Average Pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = (
        batch_size,     # batch dimension
        in_channels,    # channel dimension  
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # output dimension
    )
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x,
        out,
        batch_size,
        in_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version using Triton kernels for 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

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
        Applies 1D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)