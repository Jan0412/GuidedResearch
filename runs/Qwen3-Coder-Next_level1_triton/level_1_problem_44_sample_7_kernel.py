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
    # Calculate total number of output elements
    total_elements = batch_size * in_channels * output_length
    
    # Get current program index
    pid = tl.program_id(0)
    
    # Compute global offset
    offset = pid * BLOCK_SIZE
    block_size_actual = min(BLOCK_SIZE, total_elements - offset)
    
    # Exit early if no work to do
    if block_size_actual <= 0:
        return
    
    # Create mask for valid elements
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < total_elements
    
    # Convert linear index to (batch, channel, output_pos)
    # out_idx = batch * (in_channels * output_length) + channel * output_length + output_pos
    out_batch = indices // (in_channels * output_length)
    rest = indices % (in_channels * output_length)
    out_channel = rest // output_length
    out_pos = rest % output_length
    
    # Calculate input range for this output position
    # input_start = max(0, out_pos * stride - padding)
    # input_end = min(input_length, out_pos * stride - padding + kernel_size)
    input_start = tl.maximum(0, out_pos * stride - padding)
    input_end = tl.minimum(input_length, out_pos * stride - padding + kernel_size)
    
    # Compute effective kernel size (number of valid elements in kernel window)
    eff_kernel_size = input_end - input_start
    
    # Sum values in the pooling window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process input positions in the pooling window
    for k in range(kernel_size):
        input_pos = out_pos * stride - padding + k
        valid_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Compute input index: batch_idx * (in_channels * input_length) + channel_idx * input_length + input_pos
        input_indices = out_batch * (in_channels * input_length) + out_channel * input_length + input_pos
        input_indices = tl.where(valid_mask, input_indices, 0)  # Avoid out-of-bounds access
        
        # Load input value (will be 0 for invalid positions due to masking)
        x = tl.load(x_ptr + input_indices, mask=mask & valid_mask, other=0.0)
        
        # Accumulate sum
        sum_val = sum_val + tl.where(mask & valid_mask, x, 0.0)
    
    # Compute average (handle edge case where eff_kernel_size might be 0)
    avg = tl.where(eff_kernel_size > 0, sum_val / eff_kernel_size, 0.0)
    
    # Store result
    tl.store(out_ptr + indices, avg, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton implementation of 1D Average Pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation
        padding: Padding applied to the input
        
    Returns:
        Output tensor with shape (batch_size, in_channels, output_length)
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get input dimensions
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_elements = batch_size * in_channels * output_length
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (min(triton.cdiv(total_elements, BLOCK_SIZE), 65535),)
    
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
        Applies optimized 1D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)