import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    input_stride_batch,
    input_stride_channel,
    input_stride_seq,
    output_stride_batch,
    output_stride_channel,
    output_stride_seq,
    batch_size,
    channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and channel indices for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position in the output sequence
    output_seq_idx = tl.program_id(2) * BLOCK_SIZE
    
    # Process multiple output positions if needed
    for i in range(BLOCK_SIZE):
        if output_seq_idx + i >= output_length:
            break
            
        output_pos = output_seq_idx + i
        
        # Calculate the start and end positions in the input sequence
        start_pos = output_pos * stride - padding
        end_pos = start_pos + kernel_size
        
        # Initialize sum
        sum_val = tl.zeros([1], dtype=tl.float32)
        count = 0
        
        # Accumulate values from the pooling window
        for j in range(kernel_size):
            input_pos = start_pos + j
            # Check if the input position is valid (within bounds)
            if input_pos >= 0 and input_pos < input_length:
                # Calculate the pointer to the input element
                input_ptr_offset = (
                    batch_idx * input_stride_batch +
                    channel_idx * input_stride_channel +
                    input_pos * input_stride_seq
                )
                sum_val += tl.load(input_ptr + input_ptr_offset, mask=True)
                count += 1
                
        # Calculate average
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
            
        # Store the result
        output_ptr_offset = (
            batch_idx * output_stride_batch +
            channel_idx * output_stride_channel +
            output_pos * output_stride_seq
        )
        tl.store(output_ptr + output_ptr_offset, avg_val)

def triton_avg_pool1d(x, kernel_size, stride, padding):
    """
    Triton implementation of 1D Average Pooling
    """
    batch_size, channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_length, device=x.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Grid dimensions
    grid = (
        batch_size,      # batch dimension
        channels,        # channel dimension  
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # sequence dimension
    )
    
    # Calculate strides
    input_stride_batch, input_stride_channel, input_stride_seq = x.stride()
    output_stride_batch, output_stride_channel, output_stride_seq = output.stride()
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x,
        output,
        input_stride_batch,
        input_stride_channel,
        input_stride_seq,
        output_stride_batch,
        output_stride_channel,
        output_stride_seq,
        batch_size,
        channels,
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