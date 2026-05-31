import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,  # Input tensor pointer: (batch_size, in_channels, input_length)
    out_ptr,  # Output tensor pointer: (batch_size, in_channels, output_length)
    batch_size, in_channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute batch, channel, and output position
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_pos = tl.program_id(2)
    
    # Calculate input range for this output position
    input_start = out_pos * stride - padding
    input_end = input_start + kernel_size
    
    # Compute valid input range considering padding
    valid_start = tl.maximum(input_start, 0)
    valid_end = tl.minimum(input_end, input_length)
    
    # Calculate how many valid elements we have
    num_valid = valid_end - valid_start
    
    # Compute sum of valid elements
    sum_val = 0.0
    for i in range(valid_start, valid_end):
        x_idx = batch_idx * (in_channels * input_length) + \
                channel_idx * input_length + i
        x_val = tl.load(x_ptr + x_idx)
        sum_val += x_val
    
    # Compute average (handle case where num_valid might be 0)
    avg_val = sum_val / num_valid if num_valid > 0 else 0.0
    
    # Store result
    out_idx = batch_idx * (in_channels * output_length) + \
              channel_idx * output_length + out_pos
    tl.store(out_ptr + out_idx, avg_val)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton-based 1D average pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation
        padding: Padding applied to the input
    
    Returns:
        Output tensor with 1D average pooling applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length: floor((input_length + 2*padding - kernel_size) / stride) + 1
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Grid configuration: (batch_size, in_channels, output_length)
    grid = (batch_size, in_channels, output_length)
    
    # Launch the kernel
    avg_pool1d_kernel[grid](
        x, out, batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding, BLOCK_SIZE=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using Triton kernel.
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
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)