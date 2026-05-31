import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,               # Input tensor pointer: (batch, features, seq_len)
    y_ptr,               # Output tensor pointer: (batch, features, out_seq_len)
    batch_size,          # Batch size
    num_features,        # Number of features
    seq_len,             # Input sequence length
    out_seq_len,         # Output sequence length
    kernel_size,         # Kernel size
    stride,              # Stride
    padding,             # Padding
    dilation,            # Dilation
    BLOCK_SIZE: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    # Each program handles a portion of (batch, feature, output_position)
    # We'll use 2D grid: [num_features, out_seq_len] per batch, but we'll parallelize over batches and features
    
    batch_id = tl.program_id(0) // (num_features * out_seq_len)
    rest = tl.program_id(0) % (num_features * out_seq_len)
    feature_id = rest // out_seq_len
    out_pos = rest % out_seq_len
    
    # Compute the starting position in the input sequence
    start_input_pos = out_pos * stride - padding
    # Compute the valid kernel range
    kernel_start = tl.maximum(0, start_input_pos)
    kernel_end = tl.minimum(seq_len, start_input_pos + kernel_size * dilation)
    
    # Initialize max with -inf
    max_val = -float('inf')
    
    # Iterate over the kernel window with dilation
    # We'll process in chunks to avoid too many iterations
    current_pos = kernel_start
    while current_pos < kernel_end:
        # Load the value
        val = tl.load(
            x_ptr + batch_id * num_features * seq_len + feature_id * seq_len + current_pos,
            mask=current_pos < seq_len,
            other=-float('inf')
        )
        max_val = tl.maximum(max_val, val)
        current_pos += dilation
    
    # Store the result
    tl.store(
        y_ptr + batch_id * num_features * out_seq_len + feature_id * out_seq_len + out_pos,
        max_val
    )


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> torch.Tensor:
    """
    Applies 1D max pooling using a custom Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, sequence_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to both sides
        dilation: Spacing between kernel elements
        
    Returns:
        Output tensor with shape (batch_size, num_features, output_sequence_length)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, seq_len = x.shape
    
    # Calculate output sequence length
    out_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, num_features, out_seq_len, dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_SIZE = 1  # Not used in this implementation since we process each position individually
    # Use a 1D grid over all (batch, feature, out_pos) combinations
    total_programs = batch_size * num_features * out_seq_len
    
    # Launch kernel
    maxpool1d_kernel[total_programs,](
        x, y,
        batch_size, num_features, seq_len, out_seq_len,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_FEATURES=16,  # Not used in current implementation
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using Triton kernel.
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
        
        # Check that return_indices is False since our kernel doesn't support returning indices
        if return_indices:
            raise ValueError("Triton kernel does not support return_indices=True")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)