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
    input_len,
    output_len,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and feature index for this program
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and feature
    input_base = batch_idx * features * input_len + feature_idx * input_len
    output_base = batch_idx * features * output_len + feature_idx * output_len
    
    # Process each output element
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < output_len
    
    # Initialize max value and index
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # For each output position, compute the max over the kernel window
    for i in range(kernel_size):
        # Calculate input position for current kernel element
        input_pos = output_idx * stride + i * dilation - padding
        
        # Check if this position is valid
        valid_mask = (input_pos >= 0) & (input_pos < input_len)
        
        # Load input value
        input_val = tl.load(input_ptr + input_base + input_pos, mask=valid_mask & mask, other=float('-inf'))
        
        # Update max if current value is larger
        new_max_mask = (input_val > max_val) & valid_mask & mask
        max_val = tl.where(new_max_mask, input_val, max_val)
        max_idx = tl.where(new_max_mask, input_pos, max_idx)
    
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
        batch_size, features, sequence_length = x.shape
        
        # Calculate output length
        output_len = (sequence_length + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, features, output_len, device=x.device, dtype=torch.float32)
        
        # Prepare indices tensor if needed
        indices = None
        if self.return_indices:
            indices = torch.empty(batch_size, features, output_len, device=x.device, dtype=torch.int32)
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Grid dimensions
        grid = (
            batch_size,           # Batch dimension
            features,             # Feature dimension  
            (output_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output position dimension
        )
        
        # Launch kernel
        maxpool1d_kernel[grid](
            x,
            output,
            indices,
            batch_size,
            features,
            sequence_length,
            output_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output if not self.return_indices else (output, indices)

# Helper function to wrap the kernel call
def triton_maxpool1d(x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
    """Wrapper function for Triton-based MaxPool1D"""
    if stride is None:
        stride = kernel_size
        
    # Calculate output length
    batch_size, features, sequence_length = x.shape
    output_len = (sequence_length + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, features, output_len, device=x.device, dtype=torch.float32)
    
    # Prepare indices tensor if needed
    indices = None
    if return_indices:
        indices = torch.empty(batch_size, features, output_len, device=x.device, dtype=torch.int32)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = (
        batch_size,           # Batch dimension
        features,             # Feature dimension  
        (output_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output position dimension
    )
    
    # Launch kernel
    maxpool1d_kernel[grid](
        x,
        output,
        indices,
        batch_size,
        features,
        sequence_length,
        output_len,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output if not return_indices else (output, indices)