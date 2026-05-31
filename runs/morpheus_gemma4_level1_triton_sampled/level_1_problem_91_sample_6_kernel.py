import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    stride_outer,   # Stride of the dimension NOT being summed
    stride_inner,   # Stride of the dimension BEING summed
    n_inner,        # Size of the dimension being summed
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" (the dimension not being summed)
    outer_idx = tl.program_id(0)
    
    # Move pointers to the start of the current row
    row_x_ptr = x_ptr + outer_idx * stride_outer
    row_out_ptr = out_ptr + outer_idx * stride_outer
    
    carry = 0.0
    num_blocks = tl.cdiv(n_inner, BLOCK_SIZE)
    
    # Process blocks from the end of the dimension to the beginning
    for i in range(num_blocks - 1, -1, -1):
        # Calculate offsets for the current block
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_inner
        
        # Load the block of data
        x_block = tl.load(row_x_ptr + offsets * stride_inner, mask=mask, other=0.0)
        
        # Compute the sum of the current block
        block_sum = tl.sum(x_block)
        
        # Compute local forward cumulative sum
        # tl.cumsum is available in recent Triton versions
        local_cumsum = tl.cumsum(x_block)
        
        # Convert forward cumsum to reverse cumsum within the block:
        # reverse_cumsum[i] = block_sum - forward_cumsum[i] + x[i]
        local_rev_cumsum = block_sum - local_cumsum + x_block
        
        # Add the carry from blocks to the right
        res = local_rev_cumsum + carry
        
        # Store the result
        tl.store(row_out_ptr + offsets * stride_inner, res, mask=mask)
        
        # Update carry for the next block to the left
        carry += block_sum

def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for reverse cumulative sum along a specific dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # We assume x is a 2D tensor based on the provided architecture example.
    # For higher dimensions, we would flatten the non-summed dimensions.
    shape = x.shape
    n_inner = shape[dim]
    
    # Calculate strides
    # stride_inner: stride of the dimension we are summing over
    # stride_outer: stride of the other dimension (assuming 2D)
    # If x is (B, N) and dim=1, stride_inner = 1, stride_outer = N
    # If x is (B, N) and dim=0, stride_inner = N, stride_outer = 1
    strides = x.stride()
    stride_inner = strides[dim]
    
    # For simplicity in this 2D case, we treat the other dimension as the outer loop.
    # In a fully general case, we'd compute the product of all other dimensions.
    other_dim = 1 - dim
    n_outer = shape[other_dim]
    stride_outer = strides[other_dim]
    
    out = torch.empty_like(x)
    
    # Block size for the scan. 1024 is generally efficient for FP32.
    BLOCK_SIZE = 1024
    
    # Grid is one program per outer dimension element
    grid = (n_outer,)
    
    reverse_cumsum_kernel[grid](
        x, out, 
        stride_outer, stride_inner, 
        n_inner, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    using a custom Triton kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Replace flip -> cumsum -> flip with a single Triton kernel pass
        return triton_reverse_cumsum(x, self.dim)