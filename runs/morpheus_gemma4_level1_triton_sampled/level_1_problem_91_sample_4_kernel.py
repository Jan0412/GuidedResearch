import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rev_cumsum_kernel(
    x_ptr, 
    out_ptr, 
    stride_batch, 
    stride_reduce, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute the reverse cumulative sum along a dimension.
    
    The reverse cumulative sum R is defined as:
    R[i] = sum_{j=i}^{n-1} x[j]
    
    This can be computed using the forward cumulative sum C:
    C[i] = sum_{j=0}^{i} x[j]
    S = C[n-1] (total sum)
    R[i] = S - C[i-1] = S - (C[i] - x[i]) = S - C[i] + x[i]
    """
    # Each program handles one "row" (the batch dimension)
    batch_idx = tl.program_id(0)
    
    # Pointers to the start of the current batch row
    row_start_ptr = x_ptr + batch_idx * stride_batch
    out_start_ptr = out_ptr + batch_idx * stride_batch
    
    # Create offsets for the reduction dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the data for the current row
    # x shape: (BLOCK_SIZE,)
    x = tl.load(row_start_ptr + offsets * stride_reduce, mask=mask, other=0.0)
    
    # Compute forward cumulative sum
    # tl.cumsum is available in recent Triton versions
    c = tl.cumsum(x)
    
    # Compute the total sum of the row
    # Since x is FP32, tl.sum is efficient
    s = tl.sum(x, axis=0)
    
    # Compute reverse cumulative sum: R = S - C + x
    r = s - c + x
    
    # Store the result back to memory
    tl.store(out_start_ptr + offsets * stride_reduce, r, mask=mask)


def triton_rev_cumsum(x: torch.Tensor, dim: int):
    """
    Wrapper for the reverse cumulative sum Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Normalize dimension
    dim = dim % x.dim()
    
    # We assume x is 2D for this architecture (batch_size, input_shape)
    # Identify batch dimension and reduction dimension
    reduce_dim = dim
    batch_dim = 1 - dim
    
    batch_size = x.shape[batch_dim]
    n_elements = x.shape[reduce_dim]
    
    # Get strides to handle the layout correctly
    stride_batch = x.stride(batch_dim)
    stride_reduce = x.stride(reduce_dim)
    
    # Ensure tensor is contiguous for simpler pointer arithmetic, 
    # though we use strides for flexibility.
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Since the input shape is fixed at 32768, we use a BLOCK_SIZE of 32768.
    # In a production environment, this could be tuned or handled via multi-pass scan.
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    rev_cumsum_kernel[grid](
        x, out, 
        stride_batch, stride_reduce, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    using a custom Triton kernel to avoid multiple flip and cumsum passes.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Replace torch.cumsum(x.flip(dim), dim=dim).flip(dim) with our fused kernel
        return triton_rev_cumsum(x, self.dim)