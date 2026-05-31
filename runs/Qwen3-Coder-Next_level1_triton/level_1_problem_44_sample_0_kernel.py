import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch_idx, channel_idx) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position in the input for this (batch, channel)
    x_offset = batch_idx * in_channels * input_length + channel_idx * input_length
    
    # Output position for this (batch, channel)
    out_offset = batch_idx * in_channels * output_length + channel_idx * output_length
    
    # Process output positions in parallel using BLOCK_SIZE
    output_pos = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_pos < output_length
    
    # For each output position, compute the average over the pooling window
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate through the pooling window
    for k in range(kernel_size):
        input_pos = output_pos * stride + k - padding
        input_mask = (input_pos >= 0) & (input_pos < input_length) & output_mask
        
        # Load input values with masking
        x_indices = x_offset + input_pos
        x = tl.load(x_ptr + x_indices, mask=input_mask, other=0.0)
        
        # Accumulate sum and count for valid positions
        sum_val = tl.where(input_mask, sum_val + x, sum_val)
        count_val = tl.where(input_mask, count_val + 1.0, count_val)
    
    # Compute average (handle division by zero for empty windows)
    avg = tl.where(count_val > 0, sum_val / count_val, 0.0)
    
    # Store results
    out_indices = out_offset + output_pos
    tl.store(out_ptr + out_indices, avg, mask=output_mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Applies 1D Average Pooling using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size (int): Size of the pooling window
        stride (int): Stride of the pooling operation
        padding (int): Padding applied to the input tensor
    
    Returns:
        torch.Tensor: Output tensor with 1D Average Pooling applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length: L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    # Set block size and grid dimensions
    BLOCK_SIZE = 128
    
    # Grid: (batch_size, in_channels, ceil(output_length / BLOCK_SIZE))
    grid = (batch_size, in_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the Triton kernel
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom Triton kernel.
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
        Applies 1D Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)