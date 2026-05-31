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
    # Get the batch and feature index for this program
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and feature
    input_base = batch_idx * features * input_seq_len + feature_idx * input_seq_len
    output_base = batch_idx * features * output_seq_len + feature_idx * output_seq_len
    
    # For each output element
    for i in range(tl.program_id(2)):
        output_pos = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = output_pos < output_seq_len
        
        # Initialize max value and index
        max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
        
        # For each kernel position
        for k in range(kernel_size):
            # Calculate input position considering dilation and padding
            input_pos = output_pos * stride + k * dilation - padding
            
            # Create mask for valid positions
            valid_mask = (input_pos >= 0) & (input_pos < input_seq_len) & mask
            
            # Load input values
            input_vals = tl.load(input_ptr + input_base + input_pos, mask=valid_mask, other=float('-inf'))
            
            # Update max values and indices
            new_max = tl.maximum(max_val, input_vals)
            # Update indices when we find a new maximum
            update_mask = (input_vals > max_val) & valid_mask
            new_idx = tl.where(update_mask, input_pos, max_idx)
            
            max_val = new_max
            max_idx = new_idx
        
        # Store results
        tl.store(output_ptr + output_base + output_pos, max_val, mask=mask)
        if return_indices:
            tl.store(indices_ptr + output_base + output_pos, max_idx, mask=mask)

def triton_maxpool1d(input_tensor, kernel_size, stride, padding, dilation, return_indices=False):
    """
    Triton implementation of 1D Max Pooling
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    
    batch_size, features, seq_len = input_tensor.shape
    
    # Calculate output sequence length
    output_seq_len = (seq_len + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Allocate output tensors
    output = torch.empty(batch_size, features, output_seq_len, dtype=torch.float32, device='cuda')
    
    if return_indices:
        indices = torch.empty(batch_size, features, output_seq_len, dtype=torch.int32, device='cuda')
    else:
        indices = None
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid dimensions
    grid = (
        batch_size,      # Batch dimension
        features,        # Feature dimension  
        (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # Output sequence dimension
    )
    
    # Launch kernel
    maxpool1d_kernel[grid](
        input_tensor,
        output,
        indices,
        batch_size,
        features,
        seq_len,
        output_seq_len,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if return_indices:
        return output, indices
    else:
        return output

class ModelNew(nn.Module):
    """
    Optimized version of Max Pooling 1D using Triton kernels
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
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)

# Keep the original functions for compatibility
batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            

return_indices = False

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]