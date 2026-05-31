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
    
    # Calculate the starting position for this row
    input_row_start = row_idx * stride_input_row
    output_row_start = row_idx * stride_output_row
    
    # Process elements in chunks
    for col_start in range(0, reduction_dim_size, BLOCK_SIZE):
        # Create offsets for current chunk
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offsets < reduction_dim_size
        
        # Load input values for this chunk
        input_vals = tl.load(input_ptr + input_row_start + offsets * stride_input_col, mask=mask, other=float('inf'))
        
        # Find minimum in this chunk
        chunk_min = tl.minimum(input_vals, axis=0)
        
        # Store the minimum value
        if col_start == 0:
            # For the first chunk, initialize output
            tl.store(output_ptr + output_row_start, chunk_min, mask=tl.arange(0, 1) < 1)
        else:
            # For subsequent chunks, update existing minimum
            old_min = tl.load(output_ptr + output_row_start)
            new_min = tl.minimum(old_min, chunk_min)
            tl.store(output_ptr + output_row_start, new_min, mask=tl.arange(0, 1) < 1)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Custom Triton implementation of min reduction along a specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Make sure input is contiguous
    x = x.contiguous()
    
    # Calculate output shape
    output_shape = list(x.shape)
    del output_shape[dim]
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Handle different dimensions
    if dim == 0:
        stride_input_row = x.stride(0)
        stride_input_col = x.stride(1) if len(x.shape) > 1 else 1
        stride_output_row = out.stride(0) if len(out.shape) > 0 else 1
        stride_output_col = out.stride(1) if len(out.shape) > 1 else 1
        reduction_dim_size = x.shape[0]
    elif dim == 1:
        stride_input_row = x.stride(1) if len(x.shape) > 1 else 1
        stride_input_col = x.stride(0) if len(x.shape) > 1 else 1
        stride_output_row = out.stride(1) if len(out.shape) > 1 else 1
        stride_output_col = out.stride(0) if len(out.shape) > 0 else 1
        reduction_dim_size = x.shape[1]
    elif dim == 2:
        stride_input_row = x.stride(2) if len(x.shape) > 2 else 1
        stride_input_col = x.stride(0) if len(x.shape) > 2 else 1
        stride_output_row = out.stride(2) if len(out.shape) > 2 else 1
        stride_output_col = out.stride(0) if len(out.shape) > 0 else 1
        reduction_dim_size = x.shape[2]
    else:
        raise ValueError(f"Unsupported dimension {dim}")
    
    # Determine grid size
    num_rows = x.shape[0] if dim != 0 else 1
    if len(x.shape) > 1 and dim != 1:
        num_rows *= x.shape[1] if dim != 1 else 1
    if len(x.shape) > 2 and dim != 2:
        num_rows *= x.shape[2] if dim != 2 else 1
    
    # Adjust for case when reducing over last dimension
    if dim == len(x.shape) - 1:
        num_rows = 1
    
    # Grid configuration
    grid = (num_rows,)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    min_reduction_kernel[grid](
        x,
        out,
        x.numel(),
        stride_input_row,
        stride_input_col,
        stride_output_row,
        stride_output_col,
        reduction_dim_size,
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