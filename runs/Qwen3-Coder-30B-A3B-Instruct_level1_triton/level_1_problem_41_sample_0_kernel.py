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
    
    # Each program processes one output element
    output_idx = tl.program_id(2)
    
    if output_idx >= output_len:
        return
        
    # Calculate start position in input
    input_start = output_idx * stride - padding
    
    # Initialize max value and index
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate through kernel elements
    for i in range(kernel_size):
        # Calculate actual input position considering dilation
        pos = input_start + i * dilation
        
        # Check bounds
        if pos >= 0 and pos < input_len:
            # Load value
            val = tl.load(input_ptr + input_base + pos, mask=True, other=float('-inf'))
            
            # Update max if current value is larger
            mask = val > max_val
            max_val = tl.where(mask, val, max_val)
            max_idx = tl.where(mask, pos, max_idx)
    
    # Store result
    tl.store(output_ptr + output_base + output_idx, max_val)
    
    if return_indices:
        tl.store(indices_ptr + output_base + output_idx, max_idx)

def triton_maxpool1d(input_tensor, kernel_size, stride, padding, dilation, return_indices=False):
    """
    Triton implementation of 1D Max Pooling
    """
    batch_size, features, input_len = input_tensor.shape
    
    # Calculate output length
    output_len = (input_len + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Allocate output tensors
    output = torch.empty(batch_size, features, output_len, dtype=torch.float32, device=input_tensor.device)
    
    if return_indices:
        indices = torch.empty(batch_size, features, output_len, dtype=torch.int32, device=input_tensor.device)
    else:
        indices = None
    
    # Grid configuration
    grid = (
        batch_size,      # Batch dimension
        features,        # Feature dimension  
        output_len       # Output elements
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    maxpool1d_kernel[grid](
        input_tensor,
        output,
        indices,
        batch_size,
        features,
        input_len,
        output_len,
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