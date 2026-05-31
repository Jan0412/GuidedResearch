import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,  # Input tensor pointer: (batch, channels, input_length)
    out_ptr,  # Output tensor pointer: (batch, channels, output_length)
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    input_length,  # Input length
    output_length,  # Output length
    kernel_size,  # Size of pooling window
    stride,  # Stride of pooling operation
    padding,  # Padding applied to input
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute the starting position in the input for this output position
    # We'll process output positions in blocks for better memory access
    output_start = tl.program_id(2) * BLOCK_SIZE
    output_offsets = output_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_length
    
    # For each output position, compute average over the pooling window
    # We need to handle padding carefully
    # Effective input position = output_pos * stride - padding
    # So for each output position, we compute the valid range [start_idx, end_idx)
    # where start_idx = max(0, output_pos * stride - padding)
    #       end_idx = min(input_length, output_pos * stride - padding + kernel_size)
    
    # Vectorized computation: for each output position, compute average
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count_val = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Compute starting positions in input for each output position
    input_starts = output_offsets * stride - padding
    
    # Process each position in the pooling window
    for k in range(kernel_size):
        idx = input_starts + k
        # Check if this index is within valid input range
        valid_mask = (idx >= 0) & (idx < input_length) & output_mask
        
        # Load input values
        # We need to map (batch, channel, idx) to 1D index
        # Index = batch_idx * (channels * input_length) + channel_idx * input_length + idx
        offset = batch_idx * (in_channels * input_length) + channel_idx * input_length + idx
        
        # Load input values (with masking)
        x = tl.load(x_ptr + offset, mask=valid_mask, other=0.0)
        sum_val = sum_val + x.to(tl.float32)
        count_val = count_val + valid_mask.to(tl.int32)
    
    # Compute average (avoid division by zero)
    avg_val = sum_val / (count_val.to(tl.float32) + 1e-6)
    
    # Store result
    out_offset = batch_idx * (in_channels * output_length) + channel_idx * output_length + output_offsets
    tl.store(out_ptr + out_offset, avg_val, mask=output_mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton implementation of 1D average pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of pooling window
        stride: Stride of pooling operation
        padding: Padding applied to input
        
    Returns:
        Output tensor with shape (batch_size, in_channels, output_length)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length: floor((input_length + 2*padding - kernel_size) / stride) + 1
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 128
    
    # Grid: (batch_size, in_channels, ceil(output_length / BLOCK_SIZE))
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
    Optimized model that performs 1D Average Pooling using custom Triton kernel.
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
        Applies optimized 1D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)