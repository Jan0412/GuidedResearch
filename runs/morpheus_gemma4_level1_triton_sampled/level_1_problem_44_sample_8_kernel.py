import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr, 
    out_ptr, 
    stride, 
    padding, 
    input_length, 
    output_length, 
    BLOCK_SIZE: tl.constexpr, 
    KERNEL_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) combination and a block of the output length
    pid_bc = tl.program_id(0)  # Index for batch * channels
    pid_out = tl.program_id(1) # Index for output length block

    # Calculate the range of output indices this program is responsible for
    out_offsets = pid_out * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < output_length

    # Pointers to the start of the current row (batch, channel)
    x_row_ptr = x_ptr + pid_bc * input_length
    out_row_ptr = out_ptr + pid_bc * output_length

    # Accumulator for the sum of elements in the pooling window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the kernel window
    for k in range(KERNEL_SIZE):
        # Calculate the input index for the k-th element of the pooling window
        # window_start = j * stride - padding
        # current_index = window_start + k
        window_offsets = out_offsets * stride - padding + k
        
        # Mask to ensure indices are within the bounds of the input tensor
        in_mask = (window_offsets >= 0) & (window_offsets < input_length) & mask
        
        # Load the value (zero if out of bounds, consistent with padding=0)
        val = tl.load(x_row_ptr + window_offsets, mask=in_mask, other=0.0)
        acc += val

    # Calculate the average (PyTorch AvgPool1d by default divides by kernel_size, including padding)
    out = acc / KERNEL_SIZE
    
    # Store the result in the output tensor
    tl.store(out_row_ptr + out_offsets, out, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Wrapper for the Triton average pooling kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)
    
    # Hyperparameters for Triton
    BLOCK_SIZE = 256
    
    # Grid: (Batch * Channels, Ceiling(Output Length / BLOCK_SIZE))
    grid = (batch_size * in_channels, (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x, out, 
        stride, padding, 
        input_length, output_length, 
        BLOCK_SIZE=BLOCK_SIZE, 
        KERNEL_SIZE=kernel_size
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        # Ensure input is FP32 as requested
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)