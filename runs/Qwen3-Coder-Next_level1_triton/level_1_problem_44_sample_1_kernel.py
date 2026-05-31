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
    BLOCK_K: tl.constexpr,
):
    # Compute batch and channel indices
    bc_id = tl.program_id(0)
    batch_idx = bc_id // in_channels
    channel_idx = bc_id % in_channels
    
    # Compute output position
    out_idx = tl.program_id(1)
    out_pos = out_idx * stride - padding
    
    # Accumulator for average
    acc = 0.0
    count = 0
    
    # Iterate over kernel window in blocks
    for k_start in range(0, kernel_size, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Check if kernel positions are valid
        pos = out_pos + k_offsets
        mask = (pos >= 0) & (pos < input_length)
        # Load data with masking
        x_offsets = batch_idx * in_channels * input_length + channel_idx * input_length + pos
        x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        acc += tl.sum(x_val * mask)
        count += tl.sum(mask.to(tl.float32))
    
    # Compute average (avoid division by zero)
    avg = acc / count if count > 0 else 0.0
    
    # Store result
    out_offsets = batch_idx * in_channels * output_length + channel_idx * output_length + out_idx
    tl.store(out_ptr + out_offsets, avg)


def triton_avg_pool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    """
    Triton implementation of 1D average pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of pooling window
        stride: Stride of pooling operation
        padding: Padding applied to input
        
    Returns:
        Output tensor after average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Define kernel parameters
    BLOCK_SIZE = 128  # Not used directly in this implementation but kept for consistency
    BLOCK_K = 8       # Process up to 8 kernel positions at once
    
    # Grid: [batch_size * in_channels, output_length]
    grid = (batch_size * in_channels, output_length)
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer with Triton optimization.

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