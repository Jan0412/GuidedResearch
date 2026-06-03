import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension along which to compute
    stride_dim,  # Stride along the dimension
    other_strides,  # Strides for other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this program
    # We process one "row" perpendicular to the dimension
    row_idx = tl.program_id(0)
    
    # Calculate the base offset for this row
    base_offset = 0
    remainder = row_idx
    for i in range(len(other_strides) - 1, -1, -1):
        dim_idx = remainder // other_strides[i]
        remainder = remainder % other_strides[i]
        base_offset += dim_idx * tl.load(other_strides + i)
    
    # Process the dimension in reverse
    # We'll compute the reverse cumulative sum in-place for efficiency
    # First, load all values into registers
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # We need to handle the case where dim_size might be larger than BLOCK_SIZE
    # So we'll do it in chunks
    total_elements = dim_size
    # Create a temporary buffer to store values
    # For simplicity, we assume BLOCK_SIZE >= dim_size
    # In production code, we'd handle larger dimensions with loops
    
    # Load values in reverse order
    x_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    mask = offsets < dim_size
    rev_offsets = (dim_size - 1 - offsets) * stride_dim
    x_vals = tl.load(x_ptr + base_offset + rev_offsets, mask=mask, other=0.0)
    
    # Compute reverse cumulative sum
    cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    cumsum_val = tl.zeros((1,), dtype=tl.float32)
    for i in range(dim_size):
        val = tl.load(x_ptr + base_offset + (dim_size - 1 - i) * stride_dim, mask=(i < dim_size), other=0.0)
        cumsum_val = cumsum_val + val
        tl.store(out_ptr + base_offset + (dim_size - 1 - i) * stride_dim, cumsum_val, mask=(i < dim_size))


@triton.jit
def reverse_cumsum_kernel_v2(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (flattened other dimensions)
    dim_size,  # Size of the dimension along which to compute
    stride_row,  # Stride to move to next row
    stride_dim,  # Stride along the dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    row_offset = row_idx * stride_row
    
    # Compute reverse cumulative sum
    cumsum = tl.zeros((1,), dtype=tl.float32)
    
    # Process from end to beginning
    for i in range(dim_size - 1, -1, -1):
        offset = row_offset + i * stride_dim
        val = tl.load(x_ptr + offset)
        cumsum = cumsum + val
        tl.store(out_ptr + offset, cumsum)


class TritonReverseCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Calculate dimensions and strides
        dim = dim if dim >= 0 else x.dim() + dim
        stride_dim = x.stride(dim)
        
        # Calculate number of rows (all dimensions except dim)
        n_rows = 1
        for i, s in enumerate(x.shape):
            if i != dim:
                n_rows *= s
        
        # Calculate stride for each row
        stride_row = 1
        for i in range(dim + 1, x.dim()):
            stride_row *= x.shape[i]
        
        # Set block size based on dimension size
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = (n_rows,)
        
        # Launch kernel
        reverse_cumsum_kernel_v2[grid](
            x, out,
            n_rows, x.shape[dim],
            stride_row, stride_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For backward pass, we need to compute the gradient
        # The gradient of reverse cumulative sum is the gradient of cumulative sum
        # which is the cumulative sum of gradients in reverse order
        dim = ctx.dim
        
        # Gradient of reverse cumsum is the forward cumsum of gradients
        # Flip -> cumsum -> flip (same as original operation but for gradients)
        return torch.cumsum(grad_output.flip(dim), dim=dim).flip(dim), None


def triton_reverse_cumsum(x, dim):
    return TritonReverseCumsum.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension
    using custom Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)