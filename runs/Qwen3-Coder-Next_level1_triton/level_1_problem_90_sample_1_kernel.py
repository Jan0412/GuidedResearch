import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension along which to compute cumprod
    stride_dim,  # Stride along the dimension
    stride_other,  # Stride for other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index (outer dimension index)
    batch_idx = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_idx * stride_other
    
    # For each block along the dimension
    for start_dim in range(0, dim_size, BLOCK_SIZE):
        # Compute offsets for current block
        offsets = base_offset + start_dim * stride_dim + tl.arange(0, BLOCK_SIZE)
        mask = (start_dim + tl.arange(0, BLOCK_SIZE)) < dim_size
        
        # Load input values
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        
        # Compute cumulative product for this block
        cumprod_val = tl.cumprod(x, axis=0)
        
        # Store results
        tl.store(out_ptr + offsets, cumprod_val, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Compute cumulative product along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute cumulative product
        
    Returns:
        Tensor with same shape as x containing cumulative product
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Get dimensions and strides
    shape = x.shape
    stride = x.stride()
    
    # Calculate strides for the specified dimension
    dim_size = shape[dim]
    stride_dim = stride[dim]
    
    # Calculate stride for other dimensions (product of all dimensions except dim)
    stride_other = 1
    for i, s in enumerate(stride):
        if i != dim:
            stride_other *= shape[i] // shape[dim] if i < dim else 1
    
    # For the case where dim=1 and we have batch dimension first
    # We need to handle it properly by iterating over the batch dimension
    # and computing cumprod along dim for each batch element
    
    # Determine grid size: number of batches (all dimensions except dim)
    batch_dims = 1
    for i, s in enumerate(shape):
        if i != dim:
            batch_dims *= s
    
    # Reshape tensor to 2D for easier processing: [batch_size, dim_size]
    # where batch_size is product of all dimensions except dim
    original_shape = x.shape
    new_shape = (batch_dims, dim_size)
    
    # Get the actual strides for the 2D view
    # We need to compute stride for the dim dimension and stride for the batch dimension
    dim_stride = stride[dim]
    batch_stride = 1
    for i, s in enumerate(stride):
        if i != dim:
            batch_stride *= s
    
    BLOCK_SIZE = 128
    
    # Launch kernel with grid size equal to number of batches
    grid = (batch_dims,)
    
    cumprod_kernel[grid](
        x, out, 
        x.numel(), 
        dim_size, 
        dim_stride, 
        batch_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.
    Optimized with Triton kernel.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return triton_cumprod(x, self.dim)