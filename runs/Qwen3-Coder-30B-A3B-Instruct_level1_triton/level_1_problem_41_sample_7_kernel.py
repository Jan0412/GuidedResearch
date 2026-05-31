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
    for i in range(tl.cdiv(output_seq_len, BLOCK_SIZE)):
        output_offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = output_offset < output_seq_len
        
        # Initialize max value and index
        max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
        
        # Loop through kernel elements
        for k in range(kernel_size):
            # Calculate input position with dilation and padding
            input_pos = output_offset * stride + k * dilation - padding
            
            # Create mask for valid positions
            input_mask = (input_pos >= 0) & (input_pos < input_seq_len)
            
            # Load input values
            input_val = tl.load(input_ptr + input_base + input_pos, mask=input_mask & mask, other=float('-inf'))
            
            # Update max
            new_max_mask = input_val > max_val
            max_val = tl.where(new_max_mask, input_val, max_val)
            if return_indices:
                new_idx = tl.where(new_max_mask, input_pos, max_idx)
                max_idx = tl.where(new_max_mask, new_idx, max_idx)
        
        # Store results
        tl.store(output_ptr + output_base + output_offset, max_val, mask=mask)
        if return_indices:
            tl.store(indices_ptr + output_base + output_offset, max_idx, mask=mask)

def triton_maxpool1d(input_tensor, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
    """
    Triton implementation of 1D Max Pooling
    """
    if stride is None:
        stride = kernel_size
    
    batch_size, features, seq_len = input_tensor.shape
    
    # Calculate output length
    output_len = (seq_len + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    
    # Allocate output tensors
    output = torch.empty(batch_size, features, output_len, device=input_tensor.device, dtype=torch.float32)
    
    if return_indices:
        indices = torch.empty(batch_size, features, output_len, device=input_tensor.device, dtype=torch.int32)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = (
        batch_size,
        features,
        triton.cdiv(output_len, BLOCK_SIZE)
    )
    
    # Launch kernel
    if return_indices:
        maxpool1d_kernel[grid](
            input_tensor,
            output,
            indices,
            batch_size,
            features,
            seq_len,
            output_len,
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output, indices
    else:
        maxpool1d_kernel[grid](
            input_tensor,
            output,
            None,
            batch_size,
            features,
            seq_len,
            output_len,
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

class ModelNew(nn.Module):
    """
    Optimized version of Max Pooling 1D using Triton kernels.
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
        if self.return_indices:
            output, _ = triton_maxpool1d(
                x, 
                self.kernel_size, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.return_indices
            )
            return output
        else:
            return triton_maxpool1d(
                x, 
                self.kernel_size, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.return_indices
            )

# Input generation functions remain the same
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