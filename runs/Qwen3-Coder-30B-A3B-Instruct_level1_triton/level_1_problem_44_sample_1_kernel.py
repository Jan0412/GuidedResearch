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
    
    # Calculate the starting position in the output for this threadblock
    output_start = tl.program_id(2) * BLOCK_SIZE
    
    # Each thread processes one output element
    output_offset = output_start + tl.arange(0, BLOCK_SIZE)
    
    # Check bounds for output elements
    mask = output_offset < output_length
    
    # For each output element, compute average over the kernel window
    for i in range(BLOCK_SIZE):
        if output_start + i >= output_length:
            break
            
        output_pos = output_start + i
        if not mask[i]:
            continue
            
        # Calculate the start and end positions in the input
        input_start = output_pos * stride - padding
        input_end = input_start + kernel_size
        
        # Initialize sum accumulator
        sum_val = tl.zeros([1], dtype=tl.float32)
        count = 0
        
        # Accumulate values in the pooling window
        for j in range(kernel_size):
            input_pos = input_start + j
            # Check if the input position is valid
            if input_pos >= 0 and input_pos < input_length:
                # Calculate the global index for input tensor
                input_idx = batch_idx * (in_channels * input_length) + \
                           channel_idx * input_length + input_pos
                sum_val += tl.load(input_ptr + input_idx, mask=True)
                count += 1
                
        # Compute average
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
            
        # Calculate the global index for output tensor
        output_idx = batch_idx * (in_channels * output_length) + \
                    channel_idx * output_length + output_pos
        tl.store(output_ptr + output_idx, avg_val, mask=True)

def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0):
    """
    Triton implementation of 1D Average Pooling
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = (
        batch_size,           # Batch dimension
        in_channels,          # Channel dimension  
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output position dimension
    )
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x,
        output,
        batch_size,
        in_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for 1D Average Pooling.
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
        Applies 1D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)