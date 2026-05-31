import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr, 
    mask_ptr, 
    out_ptr,
    stride_outer, 
    stride_inner,
    n_inner,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for masked cumulative sum.
    Each program handles one 'outer' slice (row if dim=1, column if dim=0).
    """
    # The index of the outer slice we are processing
    outer_idx = tl.program_id(0)
    
    # Pointers to the start of the current slice for x, mask, and output
    x_slice_ptr = x_ptr + outer_idx * stride_outer
    mask_slice_ptr = mask_ptr + outer_idx * stride_outer
    out_slice_ptr = out_ptr + outer_idx * stride_outer
    
    # Running sum to carry over between blocks
    current_sum = 0.0
    
    # Process the inner dimension in blocks
    for i in range(0, n_inner, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Boundary mask to handle cases where n_inner is not a multiple of BLOCK_SIZE
        mask_boundary = offsets < n_inner
        
        # Load values and the mask
        # We use the inner stride to navigate elements within the slice
        x = tl.load(x_slice_ptr + offsets * stride_inner, mask=mask_boundary, other=0.0)
        m = tl.load(mask_slice_ptr + offsets * stride_inner, mask=mask_boundary, other=False)
        
        # Apply the mask: element * 1.0 if mask is True, else element * 0.0
        masked_x = x * m.to(tl.float32)
        
        # Compute the cumulative sum within the block
        local_cumsum = tl.cumsum(masked_x)
        
        # Add the carry-over sum from previous blocks to the local cumsum
        global_cumsum = local_cumsum + current_sum
        
        # Store the result
        tl.store(out_slice_ptr + offsets * stride_inner, global_cumsum, mask=mask_boundary)
        
        # Update the running sum for the next block
        # The total sum of the current block is the sum of all masked elements in it
        current_sum += tl.sum(masked_x)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Wrapper for the Triton masked_cumsum kernel.
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for predictable striding
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Determine dimensions and strides based on the target dimension (dim)
    # If dim=1, we process rows (outer=dim 0, inner=dim 1)
    # If dim=0, we process columns (outer=dim 1, inner=dim 0)
    if dim == 1:
        outer_dim_size = x.shape[0]
        inner_dim_size = x.shape[1]
        stride_outer = x.stride(0)
        stride_inner = x.stride(1)
    elif dim == 0:
        outer_dim_size = x.shape[1]
        inner_dim_size = x.shape[0]
        stride_outer = x.stride(1)
        stride_inner = x.stride(0)
    else:
        # Fallback for other dimensions, though problem context implies 0 or 1
        return torch.cumsum(x * mask, dim=dim)

    out = torch.empty_like(x)
    
    # Tuning parameter: Block size for the inner dimension scan
    BLOCK_SIZE = 1024
    
    # Grid is defined by the number of independent slices (rows or columns)
    grid = (outer_dim_size,)
    
    masked_cumsum_kernel[grid](
        x, mask, out,
        stride_outer, stride_inner,
        inner_dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using a custom Triton kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return triton_masked_cumsum(x, mask, self.dim)