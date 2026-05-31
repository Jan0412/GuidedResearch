import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduce_kernel(
    input_ptr,
    output_ptr,
    stride_input,
    stride_output,
    n_elements,
    reduce_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce along the specified dimension
    # For each element in the output, we sum over the reduce_dim_size elements
    # We need to handle the reduction properly based on the strides
    output_offsets = offsets // reduce_dim_size
    output_mask = output_offsets < (n_elements // reduce_dim_size)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform reduction using a loop approach
    for i in range(reduce_dim_size):
        input_offset = offsets + i * stride_input
        input_val = tl.load(input_ptr + input_offset, mask=(offsets + i * stride_input) < n_elements, other=0.0)
        accumulator += input_val
    
    # Store result
    tl.store(output_ptr + output_offsets, accumulator, mask=output_mask)

@triton.jit
def sum_reduce_kernel_optimized(
    input_ptr,
    output_ptr,
    n_elements,
    reduce_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate how many output elements we can process
    output_elements = n_elements // reduce_dim_size
    
    # Create mask to avoid out-of-bounds access
    mask = offsets < output_elements
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # For each element in output, accumulate reduce_dim_size elements from input
    for i in range(reduce_dim_size):
        input_offset = offsets * reduce_dim_size + i
        input_val = tl.load(input_ptr + input_offset, mask=input_offset < n_elements, other=0.0)
        accumulator += input_val
    
    # Store result
    tl.store(output_ptr + offsets, accumulator, mask=mask)

def triton_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of sum reduction along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape[dim] = 1
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Flatten the tensor for easier processing
    flat_input = x.view(-1)
    flat_output = out.view(-1)
    
    # Calculate total elements and reduction dimension size
    n_elements = flat_input.numel()
    reduce_dim_size = x.shape[dim]
    
    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    sum_reduce_kernel_optimized[grid](
        flat_input,
        flat_output,
        n_elements,
        reduce_dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for sum reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum_reduce(x, self.dim)