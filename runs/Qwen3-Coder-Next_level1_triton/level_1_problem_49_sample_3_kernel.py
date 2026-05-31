import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    total_cols,  # Total number of columns after flattening other dimensions except dim
    dim_to_reduce,  # The dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and one column (i.e., one output element)
    batch_id = tl.program_id(0)
    col_id = tl.program_id(1)
    
    # Compute total stride for indexing into input
    # We assume input is stored in row-major order
    # For simplicity, we compute the offset by iterating over dimensions
    # Instead, we'll use a more flexible approach using strides
    
    # We'll compute the index into the input tensor using strides
    # However, since Triton doesn't support dynamic indexing well, we flatten
    # and use the fact that reduction is over a contiguous dimension
    
    # Compute start offset in input
    # We assume dim_to_reduce is the innermost dimension for simplicity in this implementation
    # But to be general, we'll compute the base offset per (batch, col)
    
    # Actually, let's restructure: we'll treat input as [batch_size, num_groups, reduce_dim]
    # where num_groups = total_elements / (batch_size * reduce_dim)
    # and we reduce over the last dimension (reduce_dim)
    
    # But the input may not be contiguous in that way, so let's use the stride info
    
    # However, for simplicity and performance, we'll assume the reduction dimension is the last one
    # If not, we can transpose/reshape, but let's keep it simple for now
    
    # Actually, let's handle arbitrary dimension by computing strides manually
    
    # Get the stride of each dimension
    # Note: This is a simplified version assuming contiguous memory layout
    # For full generality, we'd need to pass strides as parameters
    
    # Let's assume input is [B, D1, D2, ..., Dn], and we reduce over dim
    # Then the output is [B, D1, ..., D_{dim-1}, D_{dim+1}, ..., Dn]
    
    # We'll flatten the input to 3D: [batch_size, num_groups, reduce_dim]
    # where num_groups = total_elements / (batch_size * reduce_dim)
    
    # Compute indices in flattened view
    # batch_offset = batch_id * (num_groups * reduce_dim)
    # group_offset = col_id * reduce_dim
    # So global offset = batch_offset + group_offset
    
    # But to be general, let's compute the actual stride-based indexing
    # However, Triton doesn't support dynamic stride computation easily
    # So we'll require that the reduction dimension is contiguous
    
    # Actually, let's use a different approach: we'll reshape the input so that
    # the reduction dimension is the innermost, then do the reduction
    
    # For now, let's assume the input is already in the right layout
    # and we're reducing over the last dimension
    
    # This is a limitation, but we can handle it by reshaping in the wrapper
    
    # Compute base offset
    base_offset = batch_id * total_cols + col_id
    
    # Now we need to reduce over dim_to_reduce elements
    # Since we're reducing over the last dimension, the stride between elements is 1
    # So we can do a simple loop
    
    # Initialize max with -inf
    max_val = -tl.float32('inf')
    
    # We'll iterate over the reduction dimension
    # But since we're in Triton, we can't easily do dynamic loops
    # So we'll use unrolling with a fixed block size
    
    # Actually, let's use a different approach: we'll use a 2D grid where
    # one dimension is batch * other_dims, and the other is the reduction dimension
    # But that doesn't match the output shape
    
    # Let's go back: we'll assume the input is [batch_size, reduce_dim, other_dims]
    # and we reduce over dim=1, so output is [batch_size, other_dims]
    # This is a common case
    
    # But the problem says "reduce over a specific dimension", which could be any
    
    # Let's implement a generic reduction by flattening to 2D first in the wrapper
    
    # For now, let's assume dim_to_reduce is the last dimension
    # and input is [batch_size, ..., reduce_dim]
    
    # Compute the start pointer for this output element
    # The output index (batch_id, col_id) corresponds to input indices
    # where all dimensions except the reduction dimension are fixed
    
    # Since we're reducing over the last dimension, the stride is 1
    # So we can do:
    # start_ptr = x_ptr + batch_id * (reduce_dim * other_dims) + col_id * reduce_dim
    
    # But we don't know reduce_dim and other_dims separately
    # So let's pass them as parameters
    
    # Actually, let's redesign: we'll reshape the input in the wrapper to be 2D
    # [batch_size * other_dims, reduce_dim], then reduce over dim=1
    
    # This is the cleanest approach, so let's do that
    
    # For now, let's assume we're reducing over the last dimension of a 3D tensor
    # [batch_size, num_groups, reduce_dim]
    
    # Compute base offset for the group
    group_base = batch_id * total_cols + col_id
    
    # Now reduce over the last dimension
    for i in range(0, dim_to_reduce, REDUCE_BLOCK_SIZE):
        offsets = i + tl.arange(0, REDUCE_BLOCK_SIZE)
        mask = offsets < dim_to_reduce
        ptr = x_ptr + group_base * dim_to_reduce + offsets
        vals = tl.load(ptr, mask=mask, other=-tl.float32('inf'))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))
    
    # Store the result
    out_ptr[group_base] = max_val


def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based max reduction over a specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with the specified dimension reduced
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Compute output shape
    output_shape = list(x.shape)
    del output_shape[dim]
    
    # Reshape input to 2D: [batch_size, reduce_dim] where batch_size = product of all dims except dim
    # and reduce_dim = dim size
    batch_size = 1
    for i, s in enumerate(x.shape):
        if i != dim:
            batch_size *= s
    
    reduce_dim = x.shape[dim]
    
    # Reshape: [batch_size, reduce_dim]
    x_reshaped = x.view(batch_size, reduce_dim)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    out_flat = out.view(-1)
    
    # Parameters
    n_groups = batch_size
    reduce_dim_size = reduce_dim
    
    # Block sizes
    BLOCK_SIZE = 128
    REDUCE_BLOCK_SIZE = 256
    
    # Grid: (n_groups,)
    grid = lambda meta: (n_groups,)
    
    # Launch kernel
    max_reduction_kernel[grid](
        x_reshaped, out_flat, n_groups, 1, reduce_dim_size,
        BLOCK_SIZE=BLOCK_SIZE, REDUCE_BLOCK_SIZE=REDUCE_BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton.
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
        Applies Max reduction over the specified dimension to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max(x, self.dim)