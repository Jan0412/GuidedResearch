import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
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
    
    # For each element in the reduced dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offsets < dim_size
        
        # Load input values
        input_vals = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=-float('inf'))
        
        # Compute max along the dimension
        max_val = tl.max(input_vals)
        
        # Store result
        tl.store(output_ptr + output_base + i * stride_output_dim, max_val, mask=mask)

def triton_max_reduction(x: torch.Tensor, dim: int):
    """
    Triton-based max reduction implementation
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    # Get dimensions
    shape = x.shape
    batch_dims = shape[:dim]
    reduced_dim_size = shape[dim]
    remaining_dims = shape[dim+1:]
    
    # Calculate strides
    stride_input_batch = 1
    stride_input_dim = 1
    stride_output_batch = 1
    stride_output_dim = 1
    
    for i in range(dim-1, -1, -1):
        stride_input_batch *= shape[i]
        stride_output_batch *= shape[i] if i != dim else 1
        
    for i in range(dim+1, len(shape)):
        stride_input_dim *= shape[i]
        stride_output_dim *= shape[i]
    
    # Calculate output shape
    output_shape = list(shape)
    output_shape.pop(dim)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate total elements
    n_elements = out.numel()
    
    # Block size
    BLOCK_SIZE = 128
    
    # Grid size
    grid = (batch_dims.numel() if batch_dims else 1,)
    
    # Launch kernel
    max_reduction_kernel[grid](
        x,
        out,
        n_elements,
        reduced_dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        stride_output_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for max reduction.
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
        return triton_max_reduction(x, self.dim)