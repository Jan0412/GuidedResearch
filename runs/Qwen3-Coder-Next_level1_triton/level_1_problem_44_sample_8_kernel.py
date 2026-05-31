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
    # Parallelize over batch_size * in_channels * output_length
    # Each program handles one output element
    bc_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    # Calculate batch and channel from bc_idx
    b = bc_idx // in_channels
    c = bc_idx % in_channels
    
    # Calculate input start and end positions for this pooling window
    in_start = out_idx * stride - padding
    in_end = min(in_start + kernel_size, input_length)
    in_start = max(in_start, 0)
    
    # Compute actual kernel size (accounting for boundary conditions)
    actual_kernel_size = in_end - in_start
    
    # Accumulator for sum
    sum_val = 0.0
    
    # Iterate over the pooling window
    for i in range(in_start, in_end):
        # Compute input index
        in_idx = i
        # Calculate flat index in input tensor
        input_offset = b * in_channels * input_length + c * input_length + in_idx
        # Load and accumulate
        x = tl.load(x_ptr + input_offset)
        sum_val += x
    
    # Compute average
    avg_val = sum_val / actual_kernel_size
    
    # Store output
    out_offset = b * in_channels * output_length + c * output_length + out_idx
    tl.store(out_ptr + out_offset, avg_val)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Apply 1D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation
        padding: Padding applied to the input
    
    Returns:
        Output tensor after average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Grid dimensions: [batch_size * in_channels, output_length]
    grid = (batch_size * in_channels, output_length)
    
    # Launch the kernel
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length,
        output_length, kernel_size, stride, padding,
        BLOCK_SIZE=1,  # We're parallelizing over output elements directly
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer with custom Triton implementation.

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
        Applies 1D Average Pooling using custom Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)