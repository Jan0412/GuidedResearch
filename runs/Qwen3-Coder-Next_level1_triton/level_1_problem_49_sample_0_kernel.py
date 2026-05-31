import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns per row (dim2)
    dim: tl.constexpr,  # Dimension to reduce over (0 or 1 in our case)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the output
    row_idx = tl.program_id(0)
    
    # Compute offsets based on which dimension we're reducing
    if dim == 1:
        # Reduce along dim=1 (dim1), so each program handles one row (dim2 elements)
        row_start = row_idx * n_cols
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_row = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        # Compute max over the row
        max_val = tl.max(x_row)
        tl.store(out_ptr + row_idx, max_val)
    else:
        # Reduce along dim=0 (batch_size), so we need to access elements across rows
        # This case is more complex, but our input is (batch_size, dim1, dim2) and 
        # dim=0 reduction means we're reducing across batch_size dimension
        # For simplicity, we'll handle 3D tensors with dim=1 or dim=2 only
        # Since the original code uses torch.max(x, dim=self.dim)[0], and self.dim is passed,
        # we'll assume dim is either 1 or 2 for the 3D input (batch_size, dim1, dim2)
        # This kernel handles only dim=1 or dim=2
        pass


@triton.jit
def max_kernel_3d_dim1(
    x_ptr,  # Pointer to input tensor of shape (batch_size, dim1, dim2)
    out_ptr,  # Pointer to output tensor of shape (batch_size, dim2)
    batch_size,  # batch_size
    dim1,  # dim1
    dim2,  # dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, dim2) pair
    batch_id = tl.program_id(0) // dim2
    dim2_id = tl.program_id(0) % dim2
    
    if batch_id < batch_size:
        # Compute the starting index for this (batch, dim2) pair
        # We need to reduce over dim1, so we have dim1 elements to check
        row_start = (batch_id * dim1 * dim2) + dim2_id
        
        # Initialize max with -inf
        max_val = -float('inf')
        
        # Process in chunks of BLOCK_SIZE
        for start in range(0, dim1, BLOCK_SIZE):
            # Offsets for the current chunk
            offsets = start * dim2 + tl.arange(0, BLOCK_SIZE) * dim2
            mask = (start + tl.arange(0, BLOCK_SIZE)) < dim1
            # Load values: x_ptr[batch_id, start:start+BLOCK_SIZE, dim2_id]
            # Memory layout is row-major: [batch][dim1][dim2]
            ptrs = x_ptr + row_start + offsets
            vals = tl.load(ptrs, mask=mask, other=-float('inf'))
            # Update max
            max_val = tl.maximum(max_val, tl.max(vals))
        
        # Store result
        out_idx = batch_id * dim2 + dim2_id
        tl.store(out_ptr + out_idx, max_val)


@triton.jit
def max_kernel_3d_dim2(
    x_ptr,  # Pointer to input tensor of shape (batch_size, dim1, dim2)
    out_ptr,  # Pointer to output tensor of shape (batch_size, dim1)
    batch_size,  # batch_size
    dim1,  # dim1
    dim2,  # dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, dim1) pair
    batch_id = tl.program_id(0) // dim1
    dim1_id = tl.program_id(0) % dim1
    
    if batch_id < batch_size:
        # Compute the starting index for this (batch, dim1) pair
        # We need to reduce over dim2, so we have dim2 elements to check
        row_start = (batch_id * dim1 * dim2) + dim1_id * dim2
        
        # Initialize max with -inf
        max_val = -float('inf')
        
        # Process in chunks of BLOCK_SIZE
        for start in range(0, dim2, BLOCK_SIZE):
            # Offsets for the current chunk
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = (start + offsets) < dim2
            # Load values: x_ptr[batch_id, dim1_id, start:start+BLOCK_SIZE]
            ptrs = x_ptr + row_start + offsets
            vals = tl.load(ptrs, mask=mask, other=-float('inf'))
            # Update max
            max_val = tl.maximum(max_val, tl.max(vals))
        
        # Store result
        out_idx = batch_id * dim1 + dim1_id
        tl.store(out_ptr + out_idx, max_val)


def triton_max_3d(x: torch.Tensor, dim: int):
    """
    Triton implementation of max reduction over a specified dimension for 3D tensors.
    
    Args:
        x: Input tensor of shape (batch_size, dim1, dim2)
        dim: Dimension to reduce over (0, 1, or 2)
    
    Returns:
        Output tensor after max reduction
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim1, dim2 = x.shape
    
    # Determine output shape and size
    if dim == 0:
        out_shape = (dim1, dim2)
    elif dim == 1:
        out_shape = (batch_size, dim2)
    elif dim == 2:
        out_shape = (batch_size, dim1)
    else:
        raise ValueError(f"Invalid dimension: {dim}. Must be 0, 1, or 2.")
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Choose appropriate kernel and launch parameters
    BLOCK_SIZE = 256
    
    if dim == 1:
        grid = lambda meta: (batch_size * dim2,)
        max_kernel_3d_dim1[grid](
            x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE
        )
    elif dim == 2:
        grid = lambda meta: (batch_size * dim1,)
        max_kernel_3d_dim2[grid](
            x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Fallback to PyTorch for dim=0 or other cases
        # For dim=0, we'd need a more complex kernel that handles reduction across batch dimension
        return torch.max(x, dim=dim)[0]
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton kernels.
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
        return triton_max_3d(x, self.dim)