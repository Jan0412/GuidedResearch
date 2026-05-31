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
    for i in range(dim_size):
        # Calculate the offset for current element
        input_offset = input_base + i * stride_input_dim
        
        # Load input value
        x = tl.load(input_ptr + input_offset)
        
        # Initialize minimum value
        if i == 0:
            min_val = x
        else:
            min_val = tl.minimum(min_val, x)
    
    # Store the result
    tl.store(output_ptr + output_base, min_val)

@triton.jit
def min_reduction_kernel_fused(
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
    
    # Shared memory for reduction within block
    shared_min = tl.shared_ptr(tl.float32, BLOCK_SIZE)
    
    # Initialize minimum value
    min_val = tl.full([], float('inf'), dtype=tl.float32)
    
    # Process elements along the reduction dimension
    for i in range(dim_size):
        # Calculate the offset for current element
        input_offset = input_base + i * stride_input_dim
        
        # Load input value
        x = tl.load(input_ptr + input_offset, mask=(i < dim_size))
        
        # Update minimum
        min_val = tl.minimum(min_val, x)
    
    # Store the result
    tl.store(output_ptr + output_base, min_val)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Triton-based min reduction implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    batch_size = 1
    dim_size = shape[dim]
    
    # Calculate batch size (product of all dimensions except the one being reduced)
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Prepare output tensor
    output_shape = list(shape)
    output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate strides
    stride_input_batch = 1
    stride_input_dim = 1
    stride_output_batch = 1
    stride_output_dim = 1
    
    # Compute strides for input tensor
    for i in range(len(shape)-1, -1, -1):
        if i == dim:
            stride_input_dim = stride_input_batch
        else:
            stride_input_batch *= shape[i]
    
    # Compute strides for output tensor
    output_stride = 1
    for i in range(len(output_shape)-1, -1, -1):
        stride_output_batch = output_stride
        output_stride *= output_shape[i]
    
    # Number of elements in the batch dimension
    n_elements = batch_size
    
    # Block size for Triton kernel
    BLOCK_SIZE = 128
    
    # Grid size
    grid = (batch_size,)
    
    # Launch kernel
    min_reduction_kernel_fused[grid](
        x,
        out,
        n_elements,
        dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        stride_output_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for min reduction.
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
        Applies min reduction over the specified dimension to the input tensor
        using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)