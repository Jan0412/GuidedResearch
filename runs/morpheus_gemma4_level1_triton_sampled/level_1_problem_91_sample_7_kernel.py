import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_dim,
    n_other,
    stride_dim,
    stride_other,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "line" along the other dimension
    other_id = tl.program_id(0)
    
    # Pointer to the start of the current line
    line_ptr = x_ptr + other_id * stride_other
    out_line_ptr = out_ptr + other_id * stride_other
    
    # Cumulative sum from the end of the line to the beginning
    acc = 0.0
    
    # Calculate how many blocks we need to cover the dimension
    num_blocks = tl.cdiv(n_dim, BLOCK_SIZE)
    
    # Iterate backwards through blocks
    for i in range(num_blocks - 1, -1, -1):
        # Calculate offsets for the current block
        block_start = i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_dim
        
        # Load the chunk of data
        chunk = tl.load(line_ptr + offsets * stride_dim, mask=mask, other=0.0)
        
        # Compute the prefix sum of the chunk
        # tl.cumsum is available in modern Triton versions
        p = tl.cumsum(chunk, axis=0)
        
        # Compute the total sum of the chunk
        # tl.sum returns a scalar when axis=0 on a 1D tensor
        total_sum = tl.sum(chunk, axis=0)
        
        # Convert prefix sum to suffix sum for the block:
        # SuffixSum[j] = TotalSum - PrefixSum[j-1]
        # SuffixSum[j] = TotalSum - (PrefixSum[j] - Chunk[j])
        s = total_sum - p + chunk
        
        # Add the accumulated sum from all blocks to the right
        out_chunk = s + acc
        
        # Store the result
        tl.store(out_line_ptr + offsets * stride_dim, out_chunk, mask=mask)
        
        # Update accumulator for the next block (to the left)
        acc += total_sum

def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    # Ensure input is on CUDA
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Handle dimensions for a 2D tensor
    # n_dim: size of the dimension to sum over
    # n_other: size of the other dimension (parallelism axis)
    shape = x.shape
    if dim == 0:
        n_dim = shape[0]
        n_other = shape[1]
        stride_dim = x.stride(0)
        stride_other = x.stride(1)
    elif dim == 1:
        n_dim = shape[1]
        n_other = shape[0]
        stride_dim = x.stride(1)
        stride_other = x.stride(0)
    else:
        raise ValueError("Only dim=0 or dim=1 supported for 2D tensors in this kernel.")

    out = torch.empty_like(x)
    
    # Block size for the scan. 1024 is generally efficient for FP32.
    BLOCK_SIZE = 1024
    
    # Grid: one program per 'other' dimension
    grid = (n_other,)
    
    reverse_cumsum_kernel[grid](
        x, out, 
        n_dim, n_other, 
        stride_dim, stride_other, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    An optimized model that performs a reverse cumulative sum operation 
    along a specified dimension using a custom Triton kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Replace torch.cumsum(x.flip(dim), dim=dim).flip(dim) with Triton kernel
        return triton_reverse_cumsum(x, self.dim)