import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,  # Input tensor pointer (B, C, L_in)
    out_ptr,  # Output tensor pointer (B, C, L_out)
    n_batches,  # Batch size
    n_channels,  # Number of channels
    input_length,  # Input length
    output_length,  # Output length
    kernel_size,  # Pooling kernel size
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Input pointers for this (batch, channel)
    x_offset = batch_idx * n_channels * input_length + channel_idx * input_length
    x_ptr_batch = x_ptr + x_offset
    
    # Output pointers for this (batch, channel)
    out_offset = batch_idx * n_channels * output_length + channel_idx * output_length
    out_ptr_batch = out_ptr + out_offset
    
    # Process output positions in blocks
    output_start = tl.program_id(2) * BLOCK_SIZE
    output_offsets = output_start + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < output_length
    
    # For each output position, compute the average
    for i in range(0, BLOCK_SIZE):
        if output_offsets[i] < output_length:
            out_pos = output_offsets[i]
            
            # Calculate the input range for this output position
            input_start = out_pos * stride - padding
            
            # Calculate actual start and end indices considering padding
            start = tl.maximum(0, input_start)
            end = tl.minimum(input_length, input_start + kernel_size)
            
            # Compute sum and count for valid elements
            sum_val = 0.0
            count = 0
            
            # Iterate over the pooling window
            for k in range(start, end):
                val = tl.load(x_ptr_batch + k)
                sum_val += val
                count += 1
            
            # Compute average
            avg_val = sum_val / count
            
            # Store result
            tl.store(out_ptr_batch + out_pos, avg_val)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton implementation of 1D Average Pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation
        padding: Padding applied to the input
    
    Returns:
        Output tensor with 1D Average Pooling applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    if output_length <= 0:
        return out
    
    # Grid configuration: (batch_size, in_channels, output_blocks)
    BLOCK_SIZE = 128
    grid = (batch_size, in_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for 1D Average Pooling.
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
        Applies optimized 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)