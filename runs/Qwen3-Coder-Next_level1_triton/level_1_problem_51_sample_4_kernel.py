import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim1,
    dim2,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine which dimension we're reducing over
    if dim == 1:
        # Reduce over dim1, output has shape [batch_size, dim2]
        batch_idx = tl.program_id(0)
        dim2_idx = tl.program_id(1)
        
        # Pointers to access the data
        # For a given [batch, :, dim2_idx], we need to find argmax over dim1
        base_ptr = x_ptr + batch_idx * (dim1 * dim2) + dim2_idx
        
        # Initialize max value and index
        max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
        max_idx = tl.full((1,), 0, dtype=tl.int32)
        
        # Iterate through dim1 in blocks
        for start in range(0, dim1, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < dim1
            
            # Load values: access [batch, offset, dim2_idx]
            ptr = base_ptr + offsets * dim2
            vals = tl.load(ptr, mask=mask, other=-float('inf'))
            
            # Update max
            curr_max = tl.maximum(max_val[0], vals)
            # Get indices where we found a new maximum
            is_new_max = (vals > max_val[0]) | ((vals == max_val[0]) & (offsets < max_idx[0]))
            # Update max value
            max_val = tl.where(vals > max_val[0], vals, max_val[0])
            # Update index (keep smallest index in case of ties)
            new_idx = tl.where(vals > max_val[0], offsets, max_idx[0])
            max_idx = tl.where(is_new_max, new_idx, max_idx[0])
        
        # Store result
        out_ptr[batch_idx * dim2 + dim2_idx] = max_idx[0]
        
    elif dim == 2:
        # Reduce over dim2, output has shape [batch_size, dim1]
        batch_idx = tl.program_id(0)
        dim1_idx = tl.program_id(1)
        
        # Pointers to access the data
        # For a given [batch, dim1_idx, :], we need to find argmax over dim2
        base_ptr = x_ptr + batch_idx * (dim1 * dim2) + dim1_idx * dim2
        
        # Initialize max value and index
        max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
        max_idx = tl.full((1,), 0, dtype=tl.int32)
        
        # Iterate through dim2 in blocks
        for start in range(0, dim2, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < dim2
            
            # Load values: access [batch, dim1_idx, offset]
            ptr = base_ptr + offsets
            vals = tl.load(ptr, mask=mask, other=-float('inf'))
            
            # Update max
            curr_max = tl.maximum(max_val[0], vals)
            is_new_max = (vals > max_val[0]) | ((vals == max_val[0]) & (offsets < max_idx[0]))
            max_val = tl.where(vals > max_val[0], vals, max_val[0])
            new_idx = tl.where(vals > max_val[0], offsets, max_idx[0])
            max_idx = tl.where(is_new_max, new_idx, max_idx[0])
        
        # Store result
        out_ptr[batch_idx * dim1 + dim1_idx] = max_idx[0]


def triton_argmax(x: torch.Tensor, dim: int):
    """
    Triton-based argmax implementation.
    
    Args:
        x: Input tensor of shape [batch_size, dim1, dim2]
        dim: Dimension to perform argmax over
        
    Returns:
        Output tensor with argmax applied, with the specified dimension removed.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim1, dim2 = x.shape
    
    # Determine output shape
    if dim == 1:
        out_shape = (batch_size, dim2)
        grid = (batch_size, dim2)
    elif dim == 2:
        out_shape = (batch_size, dim1)
        grid = (batch_size, dim1)
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only dim=1 or dim=2 supported for 3D input.")
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Set block size based on the reduction dimension
    BLOCK_SIZE = 256
    
    # Launch kernel
    argmax_kernel[grid](x, out, batch_size, dim1, dim2, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return triton_argmax(x, self.dim)