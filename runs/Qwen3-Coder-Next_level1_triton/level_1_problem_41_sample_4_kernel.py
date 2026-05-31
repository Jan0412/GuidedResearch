import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool1d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    num_features,  # Number of features (channels)
    seq_len,  # Input sequence length
    output_seq_len,  # Output sequence length
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and feature indices are computed as program IDs
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)

    # Compute base pointers for this batch and feature
    # Each (batch, feature) pair processes its own sequence independently
    input_base = x_ptr + (batch_idx * num_features + feature_idx) * seq_len
    output_base = y_ptr + (batch_idx * num_features + feature_idx) * output_seq_len

    # For each output position
    for out_pos in range(0, output_seq_len, BLOCK_SIZE):
        out_offsets = out_pos + tl.arange(0, BLOCK_SIZE)
        out_mask = out_offsets < output_seq_len

        # Compute the starting input index for this output position
        # max_pool1d formula: input_start = out_pos * stride - padding
        input_start = out_offsets * stride - padding

        # We'll compute max over kernel_size elements per output position
        # For simplicity, we compute max in a loop (unrolled for small kernel_size)
        max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

        for k in range(kernel_size):
            # Compute input index for this kernel element
            input_idx = input_start + k * dilation
            in_mask = (input_idx >= 0) & (input_idx < seq_len)
            
            # Load input values (masked)
            x = tl.load(input_base + input_idx, mask=in_mask & out_mask[:, None], other=-float('inf'))
            
            # Update max
            max_val = tl.maximum(max_val, x)

        # Store the result
        tl.store(output_base + out_offsets, max_val, mask=out_mask)


def triton_max_pool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Triton implementation of MaxPool1d.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, seq_len)
        kernel_size: Size of the max pooling window
        stride: Stride of the window (default: kernel_size)
        padding: Implicit zero padding
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor after max pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, seq_len = x.shape
    
    # Compute output sequence length
    if stride is None:
        stride = kernel_size
    
    output_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, num_features, output_seq_len, device=x.device, dtype=x.dtype)
    
    # Grid configuration: one block per (batch, feature) pair
    grid = (batch_size, num_features)
    
    # Launch kernel with BLOCK_SIZE=256 (tunable)
    BLOCK_SIZE = 256
    max_pool1d_kernel[grid](
        x, out,
        batch_size, num_features, seq_len, output_seq_len,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer with Triton optimization.

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
        output = triton_max_pool1d(
            x, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
        
        if self.return_indices:
            # For now, return only the output tensor as Triton kernel doesn't return indices
            # This implementation focuses on performance for the common case (return_indices=False)
            return output, None
        return output