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
    # Each program handles one (batch, channel, output_position) triple
    # We use 1D grid for simplicity and let each program handle a slice of work
    pid = tl.program_id(0)
    
    # Calculate indices for batch, channel, and output position
    # Total number of output elements: batch_size * in_channels * output_length
    total_outputs = batch_size * in_channels * output_length
    
    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_outputs
    
    # Decode linear index into (batch_idx, channel_idx, output_pos)
    batch_idx = offsets // (in_channels * output_length)
    temp = offsets - batch_idx * (in_channels * output_length)
    channel_idx = temp // output_length
    output_pos = temp - channel_idx * output_length
    
    # Compute input start position for this output position
    input_start = output_pos * stride - padding
    
    # Accumulator for average
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Loop over kernel window
    for k in range(kernel_size):
        input_pos = input_start + k
        # Check if input position is valid (within padding bounds)
        valid_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Compute linear index for input element
        input_indices = batch_idx * (in_channels * input_length) + channel_idx * input_length + input_pos
        input_indices = tl.where(valid_mask, input_indices, 0)  # Avoid OOB access
        
        # Load input value (using mask for OOB access)
        x_val = tl.load(x_ptr + input_indices, mask=mask & valid_mask, other=0.0)
        acc += x_val.to(tl.float32)
        count += tl.where(valid_mask, 1, 0)
    
    # Compute average (handle division by zero)
    avg = tl.where(count > 0, acc / count.to(tl.float32), 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, avg.to(x_ptr.dtype.element_ty), mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Triton implementation of 1D Average Pooling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of pooling window
        stride: Stride of pooling operation
        padding: Padding applied to input
        
    Returns:
        Output tensor with shape (batch_size, in_channels, output_length)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_outputs = batch_size * in_channels * output_length
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Determine grid size
    grid = (total_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
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