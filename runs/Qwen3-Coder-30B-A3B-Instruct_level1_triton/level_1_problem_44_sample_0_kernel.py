import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_1d_kernel(
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
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position in the output tensor
    output_offset = batch_idx * in_channels * output_length + channel_idx * output_length
    
    # For each output element, compute the average
    for i in range(tl.cdiv(output_length, BLOCK_SIZE)):
        output_start = i * BLOCK_SIZE
        output_offsets = output_start + tl.arange(0, BLOCK_SIZE)
        mask = output_offsets < output_length
        
        # Calculate the input indices for this output position
        input_start = output_start * stride - padding
        input_offsets = input_start + tl.arange(0, kernel_size)
        
        # Initialize sum accumulator
        sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Compute the sum across the kernel window
        for k in range(kernel_size):
            input_offset = input_offsets + k
            # Clamp to valid input range
            clamped_input_offset = tl.where(input_offset >= 0, input_offset, 0)
            clamped_input_offset = tl.where(clamped_input_offset < input_length, 
                                          clamped_input_offset, input_length - 1)
            
            # Load input value with boundary handling
            input_val = tl.load(input_ptr + batch_idx * in_channels * input_length + 
                               channel_idx * input_length + clamped_input_offset, 
                               mask=(clamped_input_offset < input_length) & (clamped_input_offset >= 0),
                               other=0.0)
            sum_val += input_val
            
        # Compute average (divide by kernel_size)
        avg_val = sum_val / kernel_size
        
        # Store result
        tl.store(output_ptr + output_offset + output_offsets, avg_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of 1D Average Pooling using Triton kernels.
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
        
        # Ensure output tensor is on same device as input
        output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
        
        # Make sure input is contiguous
        x = x.contiguous()
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            in_channels,
        )
        
        # Launch kernel
        avg_pool_1d_kernel[grid](
            x,
            output,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            batch_size,
            in_channels,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output