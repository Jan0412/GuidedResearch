import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    x_ptr, 
    out_ptr, 
    dim_size, 
    other_size, 
    stride_dim, 
    stride_other, 
    out_stride_dim, 
    out_stride_other,
):
    """
    Triton kernel to compute exclusive cumulative sum along a specified dimension.
    Each program handles one 'line' along the dimension being summed.
    """
    # Each program handles one index along the 'other' dimension
    line_idx = tl.program_id(0)
    if line_idx >= other_size:
        return

    # Pointer to the start of the current line
    curr_x_ptr = x_ptr + line_idx * stride_other
    curr_out_ptr = out_ptr + line_idx * out_stride_other

    # Local accumulator for the cumulative sum
    acc = 0.0
    
    # We iterate sequentially along the dimension to compute the prefix sum.
    # While sequential, the parallelism is achieved across the 'other_size' dimension.
    for i in range(dim_size):
        # Load the current element
        val = tl.load(curr_x_ptr + i * stride_dim)
        # Store the current accumulator before adding the value (making it exclusive)
        tl.store(curr_out_ptr + i * out_stride_dim, acc)
        # Update accumulator for the next element
        acc += val

def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Wrapper for the Triton exclusive cumulative sum kernel.
    """
    # Handle negative dimensions
    if dim < 0:
        dim += x.dim()

    # Ensure tensors are contiguous to simplify stride calculations
    x = x.contiguous()
    out = torch.empty_like(x)

    # Assuming x is 2D as per the model architecture provided (batch_size, input_shape)
    shape = x.shape
    if len(shape) != 2:
        # Fallback for non-2D tensors if necessary, though the problem specifies 2D
        # This part can be expanded to support n-D by flattening non-dim axes.
        raise NotImplementedError("Triton kernel currently optimized for 2D tensors.")

    # Identify sizes and strides for the dimension of interest and the other dimension
    # dim=0: Sum along rows, parallelize across columns
    # dim=1: Sum along columns, parallelize across rows
    if dim == 0:
        dim_size = shape[0]
        other_size = shape[1]
        stride_dim = x.stride(0)
        stride_other = x.stride(1)
        out_stride_dim = out.stride(0)
        out_stride_other = out.stride(1)
    elif dim == 1:
        dim_size = shape[1]
        other_size = shape[0]
        stride_dim = x.stride(1)
        stride_other = x.stride(0)
        out_stride_dim = out.stride(1)
        out_stride_other = out.stride(0)
    else:
        raise ValueError(f"Unsupported dimension {dim} for 2D tensor.")

    # Launch grid: one program per line of the other dimension
    grid = (other_size,)
    
    exclusive_cumsum_kernel[grid](
        x, 
        out, 
        dim_size, 
        other_size, 
        stride_dim, 
        stride_other, 
        out_stride_dim, 
        out_stride_other
    )
    
    return out

class ModelNew(nn.Module):
    """
    An optimized model that performs an exclusive cumulative sum using Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Replace the inefficient torch.cat + torch.cumsum + slice logic 
        # with a single-pass Triton kernel.
        return triton_exclusive_cumsum(x, self.dim)