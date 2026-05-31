import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avgpool1d_kernel(
    input_ptr,
    output_ptr,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    batch_size,
    in_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this thread's output
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Only process valid output indices
    mask = output_idx < output_length
    
    # Calculate input start positions for each output element
    input_start = output_idx * stride - padding
    
    # Compute the actual pooling values
    pooled_values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Loop over kernel elements
    for i in range(kernel_size):
        input_pos = input_start + i
        # Check if input position is valid (within padded input range)
        valid_input_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Load input value if valid
        input_val = tl.load(input_ptr + batch_idx * in_channels * input_length + 
                           channel_idx * input_length + input_pos, 
                           mask=valid_input_mask & mask, other=0.0)
        
        # Accumulate values
        pooled_values += input_val
        count += tl.where(valid_input_mask & mask, 1, 0)
    
    # Avoid division by zero
    count = tl.where(count == 0, 1, count)
    
    # Calculate average
    result = pooled_values / count.to(tl.float32)
    
    # Store result
    tl.store(output_ptr + batch_idx * in_channels * output_length + 
             channel_idx * output_length + output_idx, 
             result, mask=mask)

def triton_avgpool1d(input_tensor, kernel_size, stride, padding):
    """
    Custom Triton implementation of 1D Average Pooling.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    
    batch_size, in_channels, input_length = input_tensor.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device='cuda')
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = (
        batch_size,      # Batch dimension
        in_channels,     # Channel dimension  
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output length dimension
    )
    
    # Launch kernel
    avgpool1d_kernel[grid](
        input_tensor,
        output,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        batch_size,
        in_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using custom Triton kernels for 1D Average Pooling.
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
        Applies 1D Average Pooling using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avgpool1d(x, self.kernel_size, self.stride, self.padding)