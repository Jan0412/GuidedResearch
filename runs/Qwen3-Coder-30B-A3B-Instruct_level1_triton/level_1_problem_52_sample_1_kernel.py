import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    x_data = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    
    # Initialize min_val and min_idx
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # For each element, compute argmin
    for i in range(dim_size):
        # Calculate offset for current position
        current_offset = offsets + i * stride_x_dim
        current_mask = current_offset < n_elements
        
        # Load current value
        current_val = tl.load(x_ptr + current_offset, mask=current_mask, other=float('inf'))
        
        # Update min if current value is smaller
        mask_update = (current_val < min_val) & current_mask
        min_val = tl.where(mask_update, current_val, min_val)
        min_idx = tl.where(mask_update, i, min_idx)
    
    # Store results
    tl.store(output_ptr + offsets, min_idx, mask=mask)

def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmin operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate dimensions
    shape = x.shape
    dim_size = shape[dim]
    batch_size = 1
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Calculate stride for the dimension we're reducing over
    stride_x_dim = 1
    for i in range(dim + 1, len(shape)):
        stride_x_dim *= shape[i]
    
    # Prepare output tensor
    output_shape = list(shape)
    output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=torch.int32, device=x.device)
    
    # Number of elements in output
    n_elements = batch_size
    
    # Block size
    BLOCK_SIZE = 128
    
    # Grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    argmin_kernel[grid](
        x,
        out,
        n_elements,
        dim_size,
        stride_x_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for argmin operation.
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