import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output indices pointer
    batch_size,  # Number of batches
    dim0,  # Size of dimension 0 (if dim=1, this is batch_size)
    dim1,  # Size of dimension 1 (if dim=1, this is the reduction dimension)
    dim2,  # Size of dimension 2 (if dim=1, this is the last dimension)
    dim: tl.constexpr,  # Dimension to reduce along
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate which batch we're processing
    if dim == 1:
        # We're reducing along dim=1, so output shape is [batch_size, dim2]
        batch_idx = tl.program_id(0) // dim2
        dim2_idx = tl.program_id(0) % dim2
        
        # Calculate starting offset for this output element
        offset = batch_idx * dim1 * dim2 + dim2_idx
        
        # Initialize minimum value and index
        min_val = tl.float32(1e10)  # Large value as initial minimum
        min_idx = tl.zeros(1, dtype=tl.int32)
        
        # Process elements along dimension 1
        for i in range(0, dim1, BLOCK_SIZE):
            block_size_actual = tl.minimum(BLOCK_SIZE, dim1 - i)
            # Calculate offsets for this block
            offsets = offset + i + tl.arange(0, BLOCK_SIZE)
            mask = (i + tl.arange(0, BLOCK_SIZE)) < dim1
            
            # Load values
            x = tl.load(x_ptr + offsets, mask=mask, other=1e10)
            
            # Find minimum in this block
            block_min = tl.min(x, axis=0, return_indices=True)
            block_min_val = block_min[0]
            block_min_idx = block_min[1]
            
            # Compare with global minimum
            if i == 0:
                min_val = block_min_val
                min_idx = block_min_idx
            else:
                # Update if we found a smaller value
                should_update = (block_min_val < min_val)
                min_val = tl.where(should_update, block_min_val, min_val)
                min_idx = tl.where(should_update, block_min_idx + i, min_idx)
        
        # Store result
        out_offset = batch_idx * dim2 + dim2_idx
        tl.store(out_ptr + out_offset, min_idx)
        
    elif dim == 2:
        # We're reducing along dim=2, so output shape is [batch_size, dim1]
        batch_idx = tl.program_id(0) // dim1
        dim1_idx = tl.program_id(0) % dim1
        
        # Calculate starting offset for this output element
        offset = batch_idx * dim1 * dim2 + dim1_idx * dim2
        
        # Initialize minimum value and index
        min_val = tl.float32(1e10)
        min_idx = tl.zeros(1, dtype=tl.int32)
        
        # Process elements along dimension 2
        for j in range(0, dim2, BLOCK_SIZE):
            block_size_actual = tl.minimum(BLOCK_SIZE, dim2 - j)
            # Calculate offsets for this block
            offsets = offset + j + tl.arange(0, BLOCK_SIZE)
            mask = (j + tl.arange(0, BLOCK_SIZE)) < dim2
            
            # Load values
            x = tl.load(x_ptr + offsets, mask=mask, other=1e10)
            
            # Find minimum in this block
            block_min = tl.min(x, axis=0, return_indices=True)
            block_min_val = block_min[0]
            block_min_idx = block_min[1]
            
            # Compare with global minimum
            if j == 0:
                min_val = block_min_val
                min_idx = block_min_idx
            else:
                # Update if we found a smaller value
                should_update = (block_min_val < min_val)
                min_val = tl.where(should_update, block_min_val, min_val)
                min_idx = tl.where(should_update, block_min_idx + j, min_idx)
        
        # Store result
        out_offset = batch_idx * dim1 + dim1_idx
        tl.store(out_ptr + out_offset, min_idx)


def triton_argmin(x: torch.Tensor, dim: int):
    """
    Triton implementation of argmin along specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce along
        
    Returns:
        Tensor containing indices of minimum values
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shapes
    shape = x.shape
    if len(shape) != 3:
        raise ValueError("Expected 3D tensor")
    
    batch_size, dim0, dim1 = shape[0], shape[1], shape[2]
    
    # Determine output shape
    if dim == 1:
        out_shape = (batch_size, dim1)
    elif dim == 2:
        out_shape = (batch_size, dim0)
    else:
        raise ValueError("Only dim=1 or dim=2 supported for 3D tensor")
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    if dim == 1:
        grid_size = batch_size * dim1
    elif dim == 2:
        grid_size = batch_size * dim0
    
    grid = lambda meta: (grid_size,)
    
    # Launch kernel
    argmin_kernel[grid](
        x, out,
        batch_size, dim0, dim1, dim,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using Triton kernel.
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
        Finds the index of the minimum value along the specified dimension using Triton.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)