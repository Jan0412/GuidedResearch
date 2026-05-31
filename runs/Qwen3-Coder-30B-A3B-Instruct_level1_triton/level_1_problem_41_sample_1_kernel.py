import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    batch_size,
    features,
    input_seq_len,
    output_seq_len,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index
    batch_feature_idx = tl.program_id(0)
    output_pos = tl.program_id(1)
    
    # Calculate batch and feature indices
    batch_idx = batch_feature_idx // features
    feature_idx = batch_feature_idx % features
    
    # Shared memory for local max computation
    shared_max = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shared_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Initialize max and index
    max_val = tl.full([], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([], 0, dtype=tl.int32)
    
    # Compute input positions for this output position
    start_input = output_pos * stride - padding
    end_input = start_input + (kernel_size - 1) * dilation + 1
    
    # Process kernel elements
    for i in range(0, kernel_size):
        input_pos = start_input + i * dilation
        
        # Check if input position is valid
        if input_pos >= 0 and input_pos < input_seq_len:
            # Load input value
            input_offset = batch_idx * features * input_seq_len + feature_idx * input_seq_len + input_pos
            input_val = tl.load(input_ptr + input_offset, mask=True, other=-float('inf'))
            
            # Update max if current value is larger
            if input_val > max_val:
                max_val = input_val
                max_idx = input_pos
    
    # Store results
    output_offset = batch_idx * features * output_seq_len + feature_idx * output_seq_len + output_pos
    tl.store(output_ptr + output_offset, max_val)
    
    if return_indices:
        tl.store(indices_ptr + output_offset, max_idx)

class ModelNew(nn.Module):
    """
    Optimized Max Pooling 1D using Triton kernels.
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
        batch_size, features, sequence_length = x.shape
        
        # Calculate output sequence length
        output_seq_len = (sequence_length + 2 * self.padding - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1
        
        # Ensure x is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, features, output_seq_len, device=x.device, dtype=torch.float32)
        
        if self.return_indices:
            indices = torch.empty(batch_size, features, output_seq_len, device=x.device, dtype=torch.int32)
        else:
            indices = None
            
        # Handle case where output sequence length is 0
        if output_seq_len <= 0:
            return output
            
        # Launch kernel
        grid_size = (batch_size * features, output_seq_len)
        BLOCK_SIZE = 128
        
        # Create a wrapper for the kernel launch
        maxpool1d_kernel[grid_size](
            x,
            output,
            indices,
            batch_size,
            features,
            sequence_length,
            output_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        if self.return_indices:
            return output, indices
        else:
            return output