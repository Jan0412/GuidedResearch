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
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate which batch and channel this block processes
    batch_idx = block_idx // in_channels
    channel_idx = block_idx % in_channels
    
    # Ensure we don't go out of bounds
    if batch_idx >= batch_size:
        return
        
    # Calculate the starting position in the output for this block
    output_start = batch_idx * in_channels * output_length + channel_idx * output_length
    
    # Process each output element in this block
    for i in range(tl.cdiv(output_length, BLOCK_SIZE)):
        output_offset = i * BLOCK_SIZE
        if output_offset >= output_length:
            break
            
        # Calculate the actual output index
        output_idx = output_offset
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Perform average pooling for this output element
        for k in range(kernel_size):
            # Calculate input position
            input_pos = output_idx * stride - padding + k
            
            # Check bounds
            if input_pos >= 0 and input_pos < input_length:
                # Calculate input offset
                input_offset = batch_idx * in_channels * input_length + channel_idx * input_length + input_pos
                # Load input value
                val = tl.load(input_ptr + input_offset, mask=(input_pos < input_length), other=0.0)
                acc += val
            else:
                # Add zero for padding
                acc += 0.0
                
        # Calculate average
        avg = acc / kernel_size
        
        # Store output
        output_offset = output_start + output_idx
        tl.store(output_ptr + output_offset, avg, mask=(output_idx < output_length))


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
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid size
        grid_size = batch_size * in_channels
        
        # Launch kernel
        avg_pool1d_kernel[grid_size](
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