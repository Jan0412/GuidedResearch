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
    
    # Shared memory for reduction
    shared_data = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Loop over the output elements assigned to this block
    for i in range(output_start, min(output_start + BLOCK_SIZE, output_length)):
        # Calculate the start and end indices in the input for this pooling operation
        start_idx = i * stride - padding
        end_idx = start_idx + kernel_size
        
        # Clamp indices to valid range
        start_idx = tl.maximum(start_idx, 0)
        end_idx = tl.minimum(end_idx, input_length)
        
        # Initialize sum
        pool_sum = 0.0
        count = 0
        
        # Compute the average over the pooling window
        for j in range(start_idx, end_idx):
            # Calculate global index
            idx = batch_idx * (in_channels * input_length) + channel_idx * input_length + j
            pool_sum += tl.load(input_ptr + idx, mask=(j < input_length), other=0.0)
            count += 1
            
        # Calculate average
        if count > 0:
            avg_val = pool_sum / count
        else:
            avg_val = 0.0
            
        # Write result
        output_idx = batch_idx * (in_channels * output_length) + channel_idx * output_length + i
        tl.store(output_ptr + output_idx, avg_val)


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 1D Average Pooling.
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
        
        # Ensure tensor is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        output = torch.empty(batch_size, in_channels, output_length, dtype=torch.float32, device=x.device)
        
        # Configure grid dimensions
        grid = (
            batch_size,           # batch dimension
            in_channels,          # channel dimension
            (output_length + 127) // 128,  # output length dimension (rounded up)
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
            BLOCK_SIZE=128
        )
        
        return output