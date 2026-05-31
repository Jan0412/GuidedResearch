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
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate which batch and channel this block processes
    batch_id = block_id // in_channels
    channel_id = block_id % in_channels
    
    # Ensure we don't go out of bounds
    if batch_id >= batch_size:
        return
        
    # Calculate the starting position in the input for this output element
    output_idx = batch_id * in_channels + channel_id
    
    # For each output element, compute the average over the kernel window
    for i in range(output_length):
        # Calculate input start and end positions
        input_start = i * stride - padding
        input_end = input_start + kernel_size
        
        # Initialize sum accumulator
        sum_val = tl.zeros([1], dtype=tl.float32)
        count = 0
        
        # Compute the sum over the kernel window
        for j in range(input_start, input_end):
            # Check bounds
            if j >= 0 and j < input_length:
                # Calculate the index in the flattened tensor
                idx = batch_id * (in_channels * input_length) + channel_id * input_length + j
                sum_val += tl.load(input_ptr + idx, mask=(j >= 0) & (j < input_length))
                count += 1
                
        # Compute average
        if count > 0:
            avg = sum_val / count
        else:
            avg = 0.0
            
        # Store result
        output_idx = batch_id * (in_channels * output_length) + channel_id * output_length + i
        tl.store(output_ptr + output_idx, avg)

def triton_avgpool1d(input_tensor, kernel_size, stride, padding):
    """
    Apply 1D average pooling using a custom Triton kernel.
    """
    batch_size, in_channels, input_length = input_tensor.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Calculate total number of blocks needed
    total_blocks = batch_size * in_channels
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = (total_blocks,)
    
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
    Optimized version of 1D Average Pooling using custom Triton kernels.
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
        return triton_avgpool1d(x, self.kernel_size, self.stride, self.padding)