import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,  # Input tensor pointer (batch_size, in_channels, input_length)
    out_ptr,  # Output tensor pointer (batch_size, in_channels, output_length)
    batch_size, in_channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    # Compute output indices
    out_idx = tl.program_id(0)
    if out_idx >= batch_size * in_channels * output_length:
        return
    
    # Decode output indices: batch, channel, position
    batch_idx = out_idx // (in_channels * output_length)
    rem = out_idx % (in_channels * output_length)
    channel_idx = rem // output_length
    out_pos = rem % output_length
    
    # Compute input start and end positions for this pooling window
    input_start = out_pos * stride - padding
    input_end = input_start + kernel_size
    
    # Clamp to valid input range
    actual_start = tl.maximum(0, input_start)
    actual_end = tl.minimum(input_length, input_end)
    
    # Compute sum and count of valid elements
    sum_val = 0.0
    count = 0
    
    # Iterate over input positions in the pooling window
    for i in range(actual_start, actual_end):
        # Calculate input index
        in_idx = batch_idx * (in_channels * input_length) + channel_idx * input_length + i
        sum_val += tl.load(x_ptr + in_idx)
        count += 1
    
    # Compute average
    avg = sum_val / count if count > 0 else 0.0
    
    # Store result
    out_idx_final = batch_idx * (in_channels * output_length) + channel_idx * output_length + out_pos
    tl.store(out_ptr + out_idx_final, avg)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton implementation of 1D Average Pooling.
    
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
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    n_output_elements = batch_size * in_channels * output_length
    
    # Grid configuration
    grid = (n_output_elements,)
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=1024
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for 1D Average Pooling.
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