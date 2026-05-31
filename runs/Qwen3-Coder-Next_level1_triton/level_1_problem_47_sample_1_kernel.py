import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    inner_size,  # Size of the remaining dimensions after reduction
    dim_size,  # Size of the dimension to reduce
    stride_batch,  # Stride for batch dimension
    stride_inner,  # Stride for inner dimension
    stride_dim,  # Stride for reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_id = tl.program_id(0)
    inner_id = tl.program_id(1)
    
    # Calculate base pointers
    base_ptr = x_ptr + batch_id * stride_batch + inner_id * stride_inner
    
    # Initialize accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Loop over the reduction dimension
    for dim_idx in range(0, dim_size, BLOCK_SIZE):
        offsets = dim_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load data with mask
        data = tl.load(base_ptr + offsets * stride_dim, mask=mask, other=0.0)
        sum_val += tl.sum(data, axis=0)
    
    # Store result
    out_ptr[batch_id * inner_size + inner_id] = sum_val.to(x_ptr.dtype.element_ty)


def triton_sum(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call for sum reduction.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor with reduction applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shapes and strides
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension to positive
    if dim < 0:
        dim = ndim + dim
    
    # Calculate sizes and strides for the kernel
    batch_size = 1
    for i in range(dim):
        batch_size *= shape[i]
    
    dim_size = shape[dim]
    
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    
    # Calculate strides
    strides = x.stride()
    stride_batch = strides[0] if dim > 0 else 0
    for i in range(1, dim):
        stride_batch *= strides[i]
    
    stride_inner = strides[dim + 1] if dim < ndim - 1 else 1
    for i in range(dim + 2, ndim):
        stride_inner *= strides[i]
    
    stride_dim = strides[dim]
    
    # Prepare output tensor
    output_shape = list(shape)
    output_shape[dim] = 1
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions
    grid_batch = batch_size
    grid_inner = inner_size
    
    # Determine block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Launch kernel
    sum_kernel[grid_batch, grid_inner](
        x, out,
        batch_size, inner_size, dim_size,
        stride_batch, stride_inner, stride_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using Triton kernel instead of PyTorch's native implementation.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum(x, self.dim)