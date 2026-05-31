import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and channel
    input_base = batch_idx * in_channels * input_length + channel_idx * input_length
    output_base = batch_idx * in_channels * output_length + channel_idx * output_length
    
    # For each output element
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < output_length
    
    # Calculate the starting position in the input for this pooling operation
    start_pos = output_idx * stride - padding
    
    # Initialize accumulator
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform the pooling operation
    for i in range(kernel_size):
        input_pos = start_pos + i
        # Check if the input position is valid (within padding or within actual data)
        valid_mask = (input_pos >= 0) & (input_pos < input_length)
        # Apply both masks: output position and input validity
        combined_mask = mask & valid_mask
        
        # Load input value with masking
        input_val = tl.load(input_ptr + input_base + input_pos, mask=combined_mask, other=0.0)
        sum_val += input_val
    
    # Calculate the effective kernel size (number of valid elements)
    effective_kernel_size = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for i in range(kernel_size):
        input_pos = start_pos + i
        valid_mask = (input_pos >= 0) & (input_pos < input_length)
        effective_kernel_size += tl.where(valid_mask, 1, 0)
    
    # Avoid division by zero
    effective_kernel_size = tl.where(effective_kernel_size == 0, 1, effective_kernel_size)
    
    # Compute average
    avg_val = sum_val / effective_kernel_size.to(tl.float32)
    
    # Store result
    tl.store(output_ptr + output_base + output_idx, avg_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for 1D Average Pooling.
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
        Applies 1D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        batch_size, in_channels, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid dimensions
        grid = (
            batch_size,           # Batch dimension
            in_channels,          # Channel dimension  
            (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output length dimension
        )
        
        # Launch kernel
        avg_pool1d_kernel[grid](
            x,
            output,
            batch_size,
            in_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output