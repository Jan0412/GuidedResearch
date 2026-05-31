import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Input tensor pointer (B, C, L_in)
    y_ptr,  # Output tensor pointer (B, C, L_out)
    B,  # Batch size
    C,  # Number of channels
    L_in,  # Input sequence length
    L_out,  # Output sequence length
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID represents the output position in the sequence dimension
    pid = tl.program_id(0)
    
    # Compute batch and channel indices from the program ID
    # We'll process multiple (batch, channel) pairs per block for better occupancy
    # But for simplicity and to ensure we cover all data, we use a 1D grid over L_out
    # and handle (B, C) in the kernel
    
    # For each output position, we need to compute the max over the pooling window
    # Output position in the sequence dimension
    out_seq = pid
    
    if out_seq >= L_out:
        return
    
    # Calculate the starting position in the input sequence
    # With padding, the first valid input position is at index -padding
    # But we need to account for dilation and kernel_size
    
    # The output position out_seq corresponds to the center of the pooling window
    # For max pooling: out_seq = (in_pos + padding - dilation * (kernel_size - 1)) / stride
    # So in_pos = out_seq * stride + dilation * (kernel_size - 1) - padding
    
    # Actually, for max pooling with padding and dilation:
    # The window starts at: start = out_seq * stride - padding
    # And covers positions: start, start + dilation, start + 2*dilation, ..., start + (kernel_size-1)*dilation
    
    start_pos = out_seq * stride - padding
    
    # Initialize max value to negative infinity
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for i in range(kernel_size):
        pos = start_pos + i * dilation
        # Check if position is within valid input range
        if pos >= 0 and pos < L_in:
            # For each (batch, channel) pair, compute the input value
            # We'll process in blocks for efficiency
            pass
    
    # To handle (B, C) dimensions efficiently, we'll use a 2D grid approach
    # But for simplicity in this implementation, we'll use a different strategy:
    # Process one output position at a time, and for each output position, process all (B, C) pairs
    
    # Let's restructure: use a 2D grid where:
    # - program_id(0) = output sequence position
    # - program_id(1) = batch index
    # - program_id(2) = channel index
    
    # But Triton 1D grid is simpler, so let's flatten (B, C) into one dimension
    
    # Actually, let's use a different approach with 2D grid for better performance
    # For now, let's implement a version that processes one (batch, channel) at a time
    
    # Get the current batch and channel indices
    # This requires knowing the total number of (batch, channel) combinations
    # and using division/modulo to extract indices
    
    # Since we're in a 1D grid over output positions, we need to handle (B, C) differently
    # Let's change the grid to be 2D: (L_out, B*C)
    
    # But for this implementation, let's assume we're called with a 2D grid
    # pid0 = output position, pid1 = flattened (batch, channel) index
    
    # Wait, let me rewrite this kernel with a proper 2D grid in mind
    
    # Actually, for simplicity and correctness, I'll use a 3D grid:
    # program_id(0) = output sequence position
    # program_id(1) = batch index
    # program_id(2) = channel index
    
    # But Triton doesn't directly support 3D grids in the same way, so we'll flatten
    
    # Let's use a simpler approach: process one output position per program, 
    # and use a loop to handle all (batch, channel) combinations
    
    # Since the kernel is called with grid=(L_out,), we need to handle (B, C) differently
    # Let's change the grid to grid=(L_out, B, C) for better parallelism
    
    pass  # Placeholder - need to rewrite the kernel


@triton.jit
def maxpool1d_kernel_v2(
    x_ptr,  # Input tensor pointer (B, C, L_in)
    y_ptr,  # Output tensor pointer (B, C, L_out)
    B, C, L_in, L_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_BATCH_CHANNEL: tl.constexpr,
):
    # 2D grid: (L_out, B*C)
    seq_pid = tl.program_id(0)
    bc_pid = tl.program_id(1)
    
    if seq_pid >= L_out:
        return
    
    # Calculate batch and channel from bc_pid
    batch_idx = bc_pid // C
    channel_idx = bc_pid % C
    
    if batch_idx >= B:
        return
    
    # Compute the input sequence position range for this output position
    start_pos = seq_pid * stride - padding
    
    # Initialize max value to negative infinity
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for i in range(kernel_size):
        pos = start_pos + i * dilation
        # Check if position is within valid input range
        if pos >= 0 and pos < L_in:
            # Compute the input pointer offset for this position
            # Input layout: (B, C, L_in) - row-major
            input_offset = batch_idx * C * L_in + channel_idx * L_in + pos
            val = tl.load(x_ptr + input_offset)
            max_val = tl.maximum(max_val, val)
    
    # Store the result
    output_offset = batch_idx * C * L_out + channel_idx * L_out + seq_pid
    tl.store(y_ptr + output_offset, max_val)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1
):
    """
    Applies 1D max pooling using a custom Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, sequence_length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Implicit zero padding (default: 0)
        dilation: Spacing between kernel elements (default: 1)
    
    Returns:
        Output tensor of shape (batch_size, num_features, output_sequence_length)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, L_in = x.shape
    
    if stride is None:
        stride = kernel_size
    
    # Calculate output sequence length
    # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration: (L_out, B*C) for 2D parallelism
    grid = (L_out, B * C)
    
    # Launch the kernel
    maxpool1d_kernel_v2[grid](
        x, y,
        B, C, L_in, L_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE_SEQ=1,
        BLOCK_SIZE_BATCH_CHANNEL=128
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model with Max Pooling 1D using custom Triton kernel.
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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in the Triton kernel version.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return triton_maxpool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )