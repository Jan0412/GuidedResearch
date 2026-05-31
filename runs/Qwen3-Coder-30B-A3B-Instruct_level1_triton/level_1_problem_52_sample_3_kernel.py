import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    stride_x_batch,
    stride_x_dim1,
    stride_x_dim2,
    batch_size,
    dim1,
    dim2,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Each program handles one batch
    if batch_idx >= batch_size:
        return
        
    # Calculate base offset for this batch
    batch_offset = batch_idx * stride_x_batch
    
    # For each element in the specified dimension
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < dim1 if dim == 1 else offsets < dim2
    
    # Initialize min_val and min_idx
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # Loop through the other dimensions
    if dim == 1:  # argmin along dim1
        for i in range(dim2):
            # Calculate the offset for current position
            offset = batch_offset + i * stride_x_dim2
            # Load values
            vals = tl.load(x_ptr + offset + offsets, mask=mask, other=float('inf'))
            # Update min_val and min_idx
            new_min_mask = vals < min_val
            min_val = tl.where(new_min_mask, vals, min_val)
            min_idx = tl.where(new_min_mask, offsets, min_idx)
    else:  # argmin along dim2
        for i in range(dim1):
            # Calculate the offset for current position
            offset = batch_offset + i * stride_x_dim1
            # Load values
            vals = tl.load(x_ptr + offset + offsets, mask=mask, other=float('inf'))
            # Update min_val and min_idx
            new_min_mask = vals < min_val
            min_val = tl.where(new_min_mask, vals, min_val)
            min_idx = tl.where(new_min_mask, offsets, min_idx)
    
    # Store results
    out_offsets = batch_idx * (dim1 if dim == 1 else dim2) + offsets
    tl.store(out_ptr + out_offsets, min_idx, mask=mask)

# Simplified version for better performance when we can optimize more directly
@triton.jit
def argmin_simple_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
        
    # Calculate base offset for this batch
    batch_offset = batch_idx * dim1 * dim2
    
    # Calculate the size of the dimension we're reducing over
    reduce_dim_size = dim1 if dim == 1 else dim2
    other_dim_size = dim2 if dim == 1 else dim1
    
    # Process elements in chunks
    for i in range(other_dim_size):
        # Calculate offset for current slice
        if dim == 1:
            offset = batch_offset + i
        else:
            offset = batch_offset + i * dim1
            
        # Initialize min values for this chunk
        min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
        min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
        
        # Iterate through the reduction dimension
        for j in range(reduce_dim_size):
            # Calculate actual offset
            actual_offset = offset + (j * (dim1 if dim == 1 else 1))
            
            # Load values
            if dim == 1:
                vals = tl.load(x_ptr + actual_offset, mask=j < reduce_dim_size, other=float('inf'))
            else:
                vals = tl.load(x_ptr + actual_offset, mask=j < reduce_dim_size, other=float('inf'))
                
            # Update min values
            new_min_mask = vals < min_val
            min_val = tl.where(new_min_mask, vals, min_val)
            min_idx = tl.where(new_min_mask, j, min_idx)
            
        # Store result for this slice
        out_offset = batch_idx * other_dim_size + i
        tl.store(out_ptr + out_offset, min_idx[0])

def triton_argmin(x: torch.Tensor, dim: int):
    """
    Triton implementation of argmin operation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert dim in [0, 1, 2], "Only support dimensions 0, 1, 2"
    
    x = x.contiguous()
    
    # Get dimensions
    batch_size, dim1, dim2 = x.shape
    
    # Create output tensor
    if dim == 1:
        out = torch.zeros(batch_size, dim2, dtype=torch.int32, device=x.device)
    else:  # dim == 2
        out = torch.zeros(batch_size, dim1, dtype=torch.int32, device=x.device)
    
    # Set up grid
    BLOCK_SIZE = 128
    
    if dim == 1:
        grid = (batch_size, (dim2 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    else:
        grid = (batch_size, (dim1 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    argmin_simple_kernel[grid](
        x,
        out,
        batch_size,
        dim1,
        dim2,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for argmin operation.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)