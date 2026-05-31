import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    stride_x_row,
    stride_x_col,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program handles one row
    if row_idx >= n_elements // DIM_SIZE:
        return
        
    # Calculate base pointers for this row
    row_base = row_idx * stride_x_row
    
    # Shared memory for reduction
    shared_min_val = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_min_idx = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Initialize local min
    local_min_val = tl.full([], float('inf'), dtype=tl.float32)
    local_min_idx = tl.full([], 0, dtype=tl.int32)
    
    # Process elements in chunks
    for col_offset in range(0, DIM_SIZE, BLOCK_SIZE):
        # Calculate actual column indices
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < DIM_SIZE
        
        # Load values from global memory
        x_vals = tl.load(x_ptr + row_base + col_indices * stride_x_col, mask=mask, other=float('inf'))
        
        # Find min in this chunk
        chunk_min_val = tl.min(x_vals)
        chunk_min_idx = tl.argmin(x_vals)
        
        # Update overall min
        if chunk_min_val < local_min_val:
            local_min_val = chunk_min_val
            local_min_idx = col_offset + chunk_min_idx
            
    # Store result for this row
    tl.store(out_ptr + row_idx, local_min_idx)

def triton_argmin(x: torch.Tensor, dim: int):
    """
    Triton implementation of argmin operation along a specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    dims = x.shape
    if dim < 0:
        dim = len(dims) + dim
    
    # Calculate output size
    out_shape = list(dims)
    out_shape.pop(dim)
    out_size = 1
    for s in out_shape:
        out_size *= s
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.int32, device=x.device)
    
    # Get strides
    stride_x_row = x.stride(dim)
    stride_x_col = x.stride(0) if dim == 0 else x.stride(1) if dim == 1 else x.stride(2)
    
    # Determine grid size
    grid_size = out_size
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    argmin_kernel[(grid_size,)](x, out, out_size, stride_x_row, stride_x_col, BLOCK_SIZE=BLOCK_SIZE, DIM_SIZE=dims[dim])
    
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