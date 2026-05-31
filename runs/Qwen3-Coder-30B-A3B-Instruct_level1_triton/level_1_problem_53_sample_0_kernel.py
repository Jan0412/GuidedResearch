import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
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
        # Create offsets for this block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load data from input
        input_vals = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        local_min = tl.minimum(input_vals, axis=0)
        
        # Store the minimum value (this is a simplified approach - actual implementation would need more complex logic)
        # For simplicity, we'll just compute the overall minimum in one pass
        if i == 0:
            final_min = local_min
        else:
            final_min = tl.minimum(final_min, local_min)
    
    # Store the result
    tl.store(output_ptr + output_base, final_min)

# More efficient implementation using shared memory for reduction
@triton.jit
def min_reduction_kernel_optimized(
    input_ptr,
    output_ptr,
    batch_size,
    dim_size,
    stride_input_batch,
    stride_input_dim,
    stride_output_batch,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    input_base = batch_idx * stride_input_batch
    output_base = batch_idx * stride_output_batch
    
    # Shared memory for reduction
    shared_min = tl.shared_tensor(tl.float32, BLOCK_SIZE)
    
    # Initialize accumulator
    accumulator = float('inf')
    
    # Process elements in chunks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load data
        input_vals = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=float('inf'))
        
        # Reduce within block
        block_min = tl.reduce(input_vals, axis=0, combine_fn=tl.minimum)
        
        # Update accumulator
        accumulator = tl.minimum(accumulator, block_min)
    
    # Store result
    tl.store(output_ptr + output_base, accumulator)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Triton-based min reduction kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Handle negative dimensions
    if dim < 0:
        dim = x.dim() + dim
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Get strides
    stride_input_batch = x.stride(0) if len(x.shape) > 1 else 1
    stride_input_dim = x.stride(dim) if len(x.shape) > dim else 1
    stride_output_batch = output.stride(0) if len(output.shape) > 0 else 1
    stride_output_dim = output.stride(dim) if len(output.shape) > dim else 1
    
    # Get batch size and dimension size
    batch_size = x.shape[0] if len(x.shape) > 0 else 1
    dim_size = x.shape[dim]
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid size
    grid = (batch_size,)
    
    # Launch kernel
    min_reduction_kernel_optimized[grid](
        x,
        output,
        batch_size,
        dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernels.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)