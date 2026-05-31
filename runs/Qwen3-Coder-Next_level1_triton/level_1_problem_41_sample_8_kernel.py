import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool1d_kernel(
    x_ptr,  # Input tensor pointer: (batch, channels, seq_len)
    out_ptr,  # Output tensor pointer: (batch, channels, out_seq_len)
    batch_size, num_features, input_length, output_length,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the batch and feature dimensions
    pid_b = tl.program_id(0)  # batch index
    pid_f = tl.program_id(1)  # feature index
    pid_o = tl.program_id(2)  # output sequence index
    
    # Calculate the start and end indices of the pooling window
    # For output position pid_o, the corresponding input positions are determined by:
    # start = pid_o * stride - padding
    # end = start + (kernel_size - 1) * dilation
    start_idx = pid_o * stride - padding
    end_idx = start_idx + (kernel_size - 1) * dilation
    
    # Calculate the actual valid range within the input sequence
    valid_start = tl.maximum(start_idx, 0)
    valid_end = tl.minimum(end_idx, input_length - 1)
    
    # Calculate the offset to the current batch and feature in the input tensor
    input_offset = (pid_b * num_features * input_length + 
                   pid_f * input_length)
    
    # Calculate the offset to the current output position
    output_offset = (pid_b * num_features * output_length + 
                    pid_f * output_length + pid_o)
    
    # Initialize max value with negative infinity
    max_val = -tl.float32("inf")
    
    # Iterate through the pooling window
    current_idx = valid_start
    while current_idx <= valid_end:
        # Load the value at current_idx
        current_val = tl.load(x_ptr + input_offset + current_idx)
        # Update max value
        max_val = tl.maximum(max_val, current_val)
        # Move to next index in the pooling window
        current_idx += dilation
    
    # Store the result
    tl.store(out_ptr + output_offset, max_val)


class TritonMaxPool1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, dilation):
        # Ensure input is contiguous
        x = x.contiguous()
        
        batch_size, num_features, input_length = x.shape
        
        # Calculate output sequence length
        # Formula: (input_length + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
        output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        # Create output tensor
        out = torch.empty(batch_size, num_features, output_length, dtype=x.dtype, device=x.device)
        
        # Define block size for parallelization
        BLOCK_SIZE = 128
        
        # Grid dimensions: (batch_size, num_features, output_length)
        grid = (batch_size, num_features, output_length)
        
        # Launch the kernel
        max_pool1d_kernel[grid](
            x, out, batch_size, num_features, input_length, output_length,
            kernel_size, stride, padding, dilation,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out


def triton_maxpool1d(x, kernel_size, stride, padding, dilation):
    return TritonMaxPool1d.apply(x, kernel_size, stride, padding, dilation)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        if self.return_indices:
            raise NotImplementedError("return_indices=True is not supported in this Triton implementation")
        
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)