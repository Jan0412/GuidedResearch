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
    
    # Loop over output positions
    for i in range(tl.cdiv(output_len, BLOCK_SIZE)):
        output_pos = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = output_pos < output_len
        
        # Initialize max value and index
        max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
        
        # For each kernel position, compute the max
        for k in range(kernel_size):
            # Calculate input position
            input_pos = output_pos * stride + k * dilation - padding
            
            # Check if input position is valid
            valid_mask = (input_pos >= 0) & (input_pos < input_len)
            
            # Load input values
            input_vals = tl.load(input_ptr + input_base + input_pos, mask=valid_mask & mask, other=float('-inf'))
            
            # Update max values and indices
            new_max_mask = (input_vals > max_val) & valid_mask & mask
            max_val = tl.where(new_max_mask, input_vals, max_val)
            max_idx = tl.where(new_max_mask, input_pos, max_idx)
        
        # Store results
        tl.store(output_ptr + output_base + output_pos, max_val, mask=mask)
        if return_indices:
            tl.store(indices_ptr + output_base + output_pos, max_idx, mask=mask)

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
        output = torch.empty(batch_size, features, output_len, dtype=torch.float32, device=x.device)
        
        # Prepare indices tensor if needed
        indices = None
        if self.return_indices:
            indices = torch.empty(batch_size, features, output_len, dtype=torch.int32, device=x.device)
        
        # Set up grid dimensions
        grid = (batch_size, features)
        
        # Define block size
        BLOCK_SIZE = 128
        
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
        
        # Return appropriate output
        if self.return_indices:
            return output, indices
        else:
            return output

# Helper function to calculate output length (for reference)
def calculate_output_length(input_len, kernel_size, stride, padding, dilation):
    return (input_len + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1

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