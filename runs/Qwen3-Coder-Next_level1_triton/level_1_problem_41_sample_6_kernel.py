import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Input tensor pointer (batch, features, seq_len)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    num_features,  # Number of features
    input_seq_len,  # Input sequence length
    output_seq_len,  # Output sequence length
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a portion of the output sequence for one feature
    # We'll parallelize over batch and features, then loop over sequence
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate the base offset in the input tensor
    base_offset = batch_idx * num_features * input_seq_len + feature_idx * input_seq_len
    
    # Output sequence position for this program (we'll process in chunks)
    output_seq_start = tl.program_id(2) * BLOCK_SIZE
    output_offsets = output_seq_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid output sequence positions
    mask = output_offsets < output_seq_len
    
    # For each output position, compute the max over the dilated kernel window
    # The output position i corresponds to input positions starting at:
    # start = i * stride - padding
    # and then taking kernel_size positions with step dilation
    
    # Precompute the starting input position for each output position
    start_positions = output_offsets * stride - padding
    
    # We'll compute max for each output position
    max_values = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    # Iterate over kernel positions
    for k in range(kernel_size):
        # Compute input position for this kernel element
        input_positions = start_positions + k * dilation
        
        # Check if input position is within valid range
        valid_mask = (input_positions >= 0) & (input_positions < input_seq_len)
        
        # Compute actual input offset for valid positions
        # For invalid positions, we'll load something that doesn't affect max
        safe_input_positions = tl.where(valid_mask, input_positions, 0)
        
        # Load values from input
        load_offsets = base_offset + safe_input_positions
        values = tl.load(x_ptr + load_offsets, mask=mask, other=-float('inf'))
        
        # Update max
        max_values = tl.maximum(max_values, values)
    
    # Store results
    tl.store(out_ptr + batch_idx * num_features * output_seq_len + feature_idx * output_seq_len + output_offsets, 
             max_values, mask=mask)


def triton_maxpool1d(x, kernel_size, stride, padding, dilation):
    """
    Custom Triton implementation of MaxPool1d.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, sequence_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding applied to input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor after max pooling
    """
    batch_size, num_features, input_seq_len = x.shape
    
    # Calculate output sequence length
    # For 1D pooling: output_length = floor((input_length + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    output_seq_len = (input_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, num_features, output_seq_len, dtype=x.dtype, device=x.device)
    
    if output_seq_len <= 0:
        return out
    
    # Configure kernel launch
    BLOCK_SIZE = 128
    
    # Grid: [batch_size, num_features, (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE]
    grid = (batch_size, num_features, (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Launch kernel
    maxpool1d_kernel[grid](
        x, out,
        batch_size, num_features, input_seq_len, output_seq_len,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based MaxPool1d.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the optimized Max Pooling 1D layer.

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
        Applies optimized Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)