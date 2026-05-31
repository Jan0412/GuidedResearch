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
    num_features,
    input_seq_len,
    output_seq_len,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and feature index
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and feature
    input_base = batch_idx * num_features * input_seq_len + feature_idx * input_seq_len
    output_base = batch_idx * num_features * output_seq_len + feature_idx * output_seq_len
    
    # For each output element
    for i in range(tl.program_id(2)):
        output_idx = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = output_idx < output_seq_len
        
        # Initialize max value and index
        max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
        
        # For each kernel position
        for k in range(kernel_size):
            # Calculate input index
            input_offset = (output_idx * stride - padding) + k * dilation
            input_idx = input_offset
            
            # Check if input index is valid
            valid_mask = (input_idx >= 0) & (input_idx < input_seq_len) & mask
            
            # Load input values
            input_val = tl.load(input_ptr + input_base + input_idx, mask=valid_mask, other=float('-inf'))
            
            # Update max
            new_max = tl.maximum(max_val, input_val)
            new_max_idx = tl.where(input_val > max_val, input_idx, max_idx)
            
            max_val = new_max
            max_idx = new_max_idx
        
        # Store results
        tl.store(output_ptr + output_base + output_idx, max_val, mask=mask)
        if return_indices:
            tl.store(indices_ptr + output_base + output_idx, max_idx, mask=mask)

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
        batch_size, num_features, input_seq_len = x.shape
        
        # Calculate output sequence length
        output_seq_len = (input_seq_len + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, num_features, output_seq_len, device=x.device, dtype=torch.float32)
        
        # Prepare indices tensor if needed
        indices = None
        if self.return_indices:
            indices = torch.empty(batch_size, num_features, output_seq_len, device=x.device, dtype=torch.int32)
        
        # Set up grid dimensions
        grid = (
            batch_size,           # Batch dimension
            num_features,         # Feature dimension  
            (output_seq_len + 127) // 128  # Output sequence dimension (rounded up)
        )
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Launch kernel
        maxpool1d_kernel[grid](
            x,
            output,
            indices,
            batch_size,
            num_features,
            input_seq_len,
            output_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output if not self.return_indices else (output, indices)