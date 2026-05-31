import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_input_batch,
    stride_input_dim,
    stride_output_batch,
    stride_output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    input_base = batch_idx * stride_input_batch
    output_base = batch_idx * stride_output_batch
    
    # Process each element along the reduced dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load input values
        input_vals = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=-float('inf'))
        
        # Compute maximum
        max_val = tl.max(input_vals)
        
        # Store the result
        tl.store(output_ptr + output_base + (i // BLOCK_SIZE) * stride_output_dim, max_val)

def triton_max_reduce(x: torch.Tensor, dim: int):
    """
    Custom Triton kernel for max reduction along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    batch_dims = shape[:dim]
    reduce_dim_size = shape[dim]
    remaining_dims = shape[dim+1:]
    
    # Calculate output shape
    output_shape = list(batch_dims) + list(remaining_dims)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate strides
    stride_input_batch = 1
    stride_input_dim = 1
    stride_output_batch = 1
    stride_output_dim = 1
    
    # Calculate strides for input tensor
    for i in range(len(shape) - 1, dim, -1):
        stride_input_dim *= shape[i]
    for i in range(dim - 1, -1, -1):
        stride_input_batch *= shape[i]
    
    # Calculate strides for output tensor
    for i in range(len(output_shape) - 1, -1, -1):
        stride_output_dim *= output_shape[i] if i > 0 else 1
    for i in range(len(output_shape) - 1, 0, -1):
        stride_output_batch *= output_shape[i]
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of batches
    num_batches = 1
    for i in range(len(shape) - 1, dim, -1):
        num_batches *= shape[i]
    
    # Grid configuration
    grid = (num_batches,)
    
    # Launch the Triton kernel
    max_reduce_kernel[grid](
        x, 
        out,
        n_elements,
        reduce_dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        stride_output_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for max reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max_reduce(x, self.dim)