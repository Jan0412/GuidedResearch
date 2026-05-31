import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Input tensor pointer: (batch, features, seq_len)
    out_ptr,  # Output tensor pointer: (batch, features, output_seq_len)
    n_batch,  # Number of batches
    n_features,  # Number of features
    input_seq_len,  # Input sequence length
    output_seq_len,  # Output sequence length
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one (batch, feature) pair
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    
    # Compute starting position in the output sequence for this block
    # Each block processes multiple output positions along the sequence dimension
    output_seq_start = tl.program_id(2) * BLOCK_SIZE
    output_offsets = output_seq_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_seq_len
    
    # Calculate the starting position in the input sequence
    # For max pooling, input position = output_position * stride - padding
    input_seq_start = output_offsets * stride - padding
    
    # Load and process each kernel position
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    for k in range(kernel_size):
        # Calculate input positions for this kernel element
        input_seq_pos = input_seq_start + k * dilation
        input_mask = (input_seq_pos >= 0) & (input_seq_pos < input_seq_len)
        
        # Compute linear index: batch_id * (features * input_seq_len) + feature_id * input_seq_len + input_seq_pos
        linear_offset = batch_id * (n_features * input_seq_len) + feature_id * input_seq_len + input_seq_pos
        linear_offsets = linear_offset[:, None] if input_mask.ndim > 1 else linear_offset
        
        # Load values
        if input_mask.ndim > 1:
            values = tl.load(
                x_ptr + linear_offsets,
                mask=input_mask[:, None] & (output_offsets[:, None] < output_seq_len),
                other=-float('inf')
            )
        else:
            values = tl.load(
                x_ptr + linear_offsets,
                mask=input_mask & (output_offsets < output_seq_len),
                other=-float('inf')
            )
        
        # Update maximum
        max_val = tl.maximum(max_val, values)
    
    # Store results
    out_linear_offsets = batch_id * (n_features * output_seq_len) + feature_id * output_seq_len + output_offsets
    tl.store(
        out_ptr + out_linear_offsets,
        max_val,
        mask=output_mask
    )


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int = None, 
                    padding: int = 0, dilation: int = 1, return_indices: bool = False):
    """
    Applies 1D max pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, sequence_length)
        kernel_size, stride, padding, dilation, return_indices: Same as torch.nn.MaxPool1d
        
    Returns:
        Output tensor of shape (batch_size, num_features, output_sequence_length)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, input_seq_len = x.shape
    
    # Calculate output sequence length
    if stride is None:
        stride = kernel_size
        
    output_seq_len = (input_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, num_features, output_seq_len, dtype=x.dtype, device=x.device)
    
    if return_indices:
        raise NotImplementedError("return_indices=True is not implemented in Triton kernel")
    
    # Grid dimensions: (batch_size, num_features, (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    BLOCK_SIZE = 128
    grid = (batch_size, num_features, (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    maxpool1d_kernel[grid](
        x, out,
        batch_size, num_features,
        input_seq_len, output_seq_len,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer with custom Triton implementation.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
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
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, 
                               self.dilation, self.return_indices)