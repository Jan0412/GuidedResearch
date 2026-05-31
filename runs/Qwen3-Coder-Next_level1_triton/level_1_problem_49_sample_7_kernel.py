import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of columns in the reduction dimension
    stride_batch,  # Stride between batches
    stride_rows,  # Stride between rows in each batch
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and row indices
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    # Compute base pointer for this (batch, row) position
    # Assuming x is laid out as [batch_size, dim1, dim2] and we're reducing over dim=self.dim
    # For dim=1 (dim1), we're reducing over columns of shape [dim1, dim2]
    # For dim=2 (dim2), we're reducing over innermost dimension
    base_ptr = x_ptr + batch_idx * stride_batch + row_idx * stride_rows
    
    # Initialize maximum with very small value
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    # Iterate over the reduction dimension in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data with masking
        data = tl.load(base_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Update maximum
        max_val = tl.maximum(max_val, data)
    
    # Perform reduction within the block to get maximum
    # Use a tree reduction
    for i in range(BLOCK_SIZE // 2):
        max_val = tl.maximum(max_val, tl.roll(max_val, 1 << i))
    
    # Store result only for thread 0
    if tl.program_id(0) == 0 and tl.program_id(1) == 0 and tl.program_id(2) == 0:
        # For simplicity, we'll use a separate kernel for each dimension case
        # This kernel will be specialized for each dimension
        pass


@triton.jit
def max_reduction_dim1_kernel(
    x_ptr,  # Pointer to input tensor [batch_size, dim1, dim2]
    out_ptr,  # Pointer to output tensor [batch_size, dim2]
    batch_size,  # batch_size
    dim1,  # dim1
    dim2,  # dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one (batch, dim2) pair
    batch_idx = tl.program_id(0)
    dim2_idx = tl.program_id(1)
    
    # Compute base pointer for this (batch, dim2) position
    base_ptr = x_ptr + batch_idx * dim1 * dim2 + dim2_idx
    
    # Initialize maximum with very small value
    max_val = -float('inf')
    
    # Iterate over dim1 dimension
    for start in range(0, dim1, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        
        # Compute pointer offsets for this dim1 index
        ptrs = base_ptr + offsets * dim2
        
        # Load data with masking
        data = tl.load(ptrs, mask=mask, other=-float('inf'))
        
        # Update maximum
        max_val = tl.maximum(max_val, data)
    
    # Store result
    out_ptr_idx = batch_idx * dim2 + dim2_idx
    tl.store(out_ptr + out_ptr_idx, max_val)


@triton.jit
def max_reduction_dim2_kernel(
    x_ptr,  # Pointer to input tensor [batch_size, dim1, dim2]
    out_ptr,  # Pointer to output tensor [batch_size, dim1]
    batch_size,  # batch_size
    dim1,  # dim1
    dim2,  # dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one (batch, dim1) pair
    batch_idx = tl.program_id(0)
    dim1_idx = tl.program_id(1)
    
    # Compute base pointer for this (batch, dim1) position
    base_ptr = x_ptr + batch_idx * dim1 * dim2 + dim1_idx * dim2
    
    # Initialize maximum with very small value
    max_val = -float('inf')
    
    # Iterate over dim2 dimension
    for start in range(0, dim2, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim2
        
        # Load data with masking
        data = tl.load(base_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Update maximum
        max_val = tl.maximum(max_val, data)
    
    # Store result
    out_ptr_idx = batch_idx * dim1 + dim1_idx
    tl.store(out_ptr + out_ptr_idx, max_val)


def triton_max_reduction(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call for max reduction.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    shape = list(x.shape)
    shape.pop(dim)
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    
    # Get dimensions
    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]
    
    BLOCK_SIZE = 256
    
    if dim == 1:
        # Reduce over dim1
        grid = (batch_size, dim2)
        max_reduction_dim1_kernel[grid](
            x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE
        )
    elif dim == 2:
        # Reduce over dim2
        grid = (batch_size, dim1)
        max_reduction_dim2_kernel[grid](
            x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError("Only dim=1 or dim=2 are supported for this kernel")
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton kernel.
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
        Applies Max reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max_reduction(x, self.dim)


# Set the dimensions for the model
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension