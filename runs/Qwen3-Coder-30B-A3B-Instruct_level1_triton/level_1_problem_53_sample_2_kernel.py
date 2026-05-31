import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_row,
    stride_input_col,
    stride_output_row,
    stride_output_col,
    reduction_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Calculate total elements per row
    elements_per_row = reduction_dim_size
    
    # Shared memory for reduction
    shared_data = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize minimum value
    min_val = tl.full([], float('inf'), dtype=tl.float32)
    
    # Loop through all columns in this row
    for col_start in range(0, elements_per_row, BLOCK_SIZE):
        # Calculate actual column indices
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < elements_per_row
        
        # Load data from global memory
        input_offsets = row_idx * stride_input_row + col_offsets * stride_input_col
        input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=float('inf'))
        
        # Reduce within block
        block_min = tl.minimum(input_vals, axis=0)
        shared_data[tl.arange(0, BLOCK_SIZE)] = block_min
        
        # Synchronize threads
        tl.sync()
        
        # Reduce across the block
        if BLOCK_SIZE >= 256:
            if tl.thread_id() < 128:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 128])
            tl.sync()
        if BLOCK_SIZE >= 128:
            if tl.thread_id() < 64:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 64])
            tl.sync()
        if BLOCK_SIZE >= 64:
            if tl.thread_id() < 32:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 32])
            tl.sync()
        if BLOCK_SIZE >= 32:
            if tl.thread_id() < 16:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 16])
            tl.sync()
        if BLOCK_SIZE >= 16:
            if tl.thread_id() < 8:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 8])
            tl.sync()
        if BLOCK_SIZE >= 8:
            if tl.thread_id() < 4:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 4])
            tl.sync()
        if BLOCK_SIZE >= 4:
            if tl.thread_id() < 2:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 2])
            tl.sync()
        if BLOCK_SIZE >= 2:
            if tl.thread_id() < 1:
                shared_data[tl.thread_id()] = tl.minimum(shared_data[tl.thread_id()], shared_data[tl.thread_id() + 1])
            tl.sync()
        
        # Update global minimum
        min_val = tl.minimum(min_val, shared_data[0])
    
    # Write result
    output_offset = row_idx * stride_output_row
    tl.store(output_ptr + output_offset, min_val)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Triton implementation of min reduction along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    
    # Handle different dimensions
    if dim == 0:
        # Reduction along first dimension
        num_rows = x.shape[0]
        elements_per_row = x.shape[1] * x.shape[2]  # Assuming 3D case
        stride_input_row = x.stride(0)
        stride_input_col = x.stride(1) if len(x.shape) > 1 else 1
        stride_output_row = 1
        stride_output_col = 1
    elif dim == 1:
        # Reduction along second dimension  
        num_rows = x.shape[0]
        elements_per_row = x.shape[2]
        stride_input_row = x.stride(1)
        stride_input_col = x.stride(2)
        stride_output_row = x.stride(0)
        stride_output_col = 1
    elif dim == 2:
        # Reduction along third dimension
        num_rows = x.shape[0] * x.shape[1]
        elements_per_row = x.shape[2]
        stride_input_row = x.stride(1) if len(x.shape) > 1 else x.stride(0)
        stride_input_col = x.stride(2)
        stride_output_row = x.stride(0)
        stride_output_col = x.stride(1) if len(x.shape) > 1 else 1
    else:
        raise ValueError(f"Unsupported dimension {dim}")
    
    # Grid configuration
    BLOCK_SIZE = 256
    grid = (num_rows,)
    
    # Launch kernel
    min_reduction_kernel[grid](
        x,
        out,
        x.numel(),
        stride_input_row,
        stride_input_col,
        stride_output_row,
        stride_output_col,
        elements_per_row,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for min reduction.
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
        Applies min reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)