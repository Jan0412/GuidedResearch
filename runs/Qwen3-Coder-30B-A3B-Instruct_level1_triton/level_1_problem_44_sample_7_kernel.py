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
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate which batch and channel this block processes
    batch_id = block_id // in_channels
    channel_id = block_id % in_channels
    
    # Early exit if out of bounds
    if batch_id >= batch_size:
        return
        
    # Calculate the starting position in the output
    output_idx = batch_id * in_channels * output_length + channel_id * output_length
    
    # Process each output element in this block
    for i in range(tl.cdiv(output_length, BLOCK_SIZE)):
        output_offset = i * BLOCK_SIZE
        output_pos = output_offset + tl.arange(0, BLOCK_SIZE)
        
        # Mask for valid output positions
        valid_output = output_pos < output_length
        
        # Calculate input positions for the pooling window
        input_start = output_pos * stride - padding
        input_end = input_start + kernel_size
        
        # Initialize sum accumulator
        sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Compute average over the pooling window
        for k in range(kernel_size):
            input_pos = input_start + k
            # Check bounds for input
            valid_input = (input_pos >= 0) & (input_pos < input_length)
            
            # Calculate input index
            input_idx = batch_id * in_channels * input_length + channel_id * input_length + input_pos
            
            # Load input value if valid
            input_val = tl.where(valid_input, 
                               tl.load(input_ptr + input_idx, mask=valid_input, other=0.0),
                               0.0)
            
            # Accumulate
            sum_val += input_val
        
        # Compute average (divide by kernel_size)
        avg_val = sum_val / kernel_size
        
        # Store output
        output_idx_local = output_idx + output_pos
        tl.store(output_ptr + output_idx_local, avg_val, mask=valid_output)


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
        
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
        
        # Calculate total number of blocks needed
        total_blocks = batch_size * in_channels
        
        # Set block size for Triton kernel
        BLOCK_SIZE = 128
        
        # Define grid size
        grid = lambda meta: (total_blocks,)
        
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