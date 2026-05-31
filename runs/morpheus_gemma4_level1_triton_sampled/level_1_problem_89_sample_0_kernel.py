import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def local_scan_kernel(
    x_ptr, out_ptr, block_sums_ptr,
    N, 
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr
):
    # Row and block indices
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N - block_id * BLOCK_SIZE)
    
    # Load input values
    ptr = x_ptr + row * N + block_id * BLOCK_SIZE + offsets
    vals = tl.load(ptr, mask=mask, other=0.0)
    
    # Compute local cumulative sum within the block
    res = tl.cumsum(vals)
    
    # Store the local cumulative sum
    tl.store(out_ptr + row * N + block_id * BLOCK_SIZE + offsets, res, mask=mask)
    
    # Store the total sum of this block for the global scan
    # We use tl.sum to get the total sum of the elements in this block
    block_total = tl.sum(vals)
    tl.store(block_sums_ptr + row * MAX_BLOCKS + block_id, block_total)

@triton.jit
def block_sum_scan_kernel(
    block_sums_ptr, prefix_sums_ptr, 
    NUM_BLOCKS, 
    MAX_BLOCKS: tl.constexpr
):
    # Each program handles one row
    row = tl.program_id(0)
    
    # Offsets for the block sums
    offsets = tl.arange(0, MAX_BLOCKS)
    mask = offsets < NUM_BLOCKS
    
    # Load the block sums for this row
    sums = tl.load(block_sums_ptr + row * MAX_BLOCKS + offsets, mask=mask, other=0.0)
    
    # Compute the cumulative sum of the block sums
    res = tl.cumsum(sums)
    
    # Store the prefix sums
    tl.store(prefix_sums_ptr + row * MAX_BLOCKS + offsets, res, mask=mask)

@triton.jit
def global_add_kernel(
    out_ptr, prefix_sums_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr
):
    # Row and block indices
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N - block_id * BLOCK_SIZE)
    
    # Load the prefix sum of all previous blocks
    prev_sum = 0.0
    if block_id > 0:
        prev_sum = tl.load(prefix_sums_ptr + row * MAX_BLOCKS + block_id - 1)
    
    # Load, add the offset, and store
    ptr = out_ptr + row * N + block_id * BLOCK_SIZE + offsets
    vals = tl.load(ptr, mask=mask, other=0.0)
    tl.store(ptr, vals + prev_sum, mask=mask)

def triton_cumsum(x: torch.Tensor, dim: int):
    # Handle negative dimensions
    dim = dim % x.ndim
    
    # Permute the tensor so that the target dimension is the last one
    dims = list(range(x.ndim))
    dims.pop(dim)
    dims.append(dim)
    x_permuted = x.permute(*dims).contiguous()
    
    # Reshape to (B, N) where N is the size of the dimension to sum over
    original_shape = x_permuted.shape
    N = original_shape[-1]
    B = x_permuted.numel() // N
    x_reshaped = x_permuted.reshape(B, N)
    
    # Hyperparameters
    BLOCK_SIZE = 1024
    MAX_BLOCKS = 1024 # Supports N up to 1024 * 1024
    NUM_BLOCKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output and auxiliary tensors
    out = torch.empty_like(x_reshaped)
    block_sums = torch.zeros((B, MAX_BLOCKS), device=x.device, dtype=torch.float32)
    prefix_sums = torch.zeros((B, MAX_BLOCKS), device=x.device, dtype=torch.float32)
    
    # Pass 1: Local scan within blocks
    grid1 = (B, NUM_BLOCKS)
    local_scan_kernel[grid1](
        x_reshaped, out, block_sums,
        N, BLOCK_SIZE=BLOCK_SIZE, MAX_BLOCKS=MAX_BLOCKS
    )
    
    # Pass 2: Scan of block sums
    grid2 = (B,)
    block_sum_scan_kernel[grid2](
        block_sums, prefix_sums,
        NUM_BLOCKS, MAX_BLOCKS=MAX_BLOCKS
    )
    
    # Pass 3: Global offset addition
    grid3 = (B, NUM_BLOCKS)
    global_add_kernel[grid3](
        out, prefix_sums,
        N, BLOCK_SIZE=BLOCK_SIZE, MAX_BLOCKS=MAX_BLOCKS
    )
    
    # Reshape and permute back to original layout
    out_reshaped = out.reshape(original_shape)
    
    # Inverse permutation
    inv_dims = [0] * x.ndim
    for i, d in enumerate(dims):
        inv_dims[d] = i
        
    return out_reshaped.permute(*inv_dims).contiguous()

class ModelNew(nn.Module):
    """
    An optimized model that performs a cumulative sum (prefix sum) operation 
    along a specified dimension using custom Triton kernels.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Ensure input is FP32 for the Triton kernel
        original_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        result = triton_cumsum(x, self.dim)
        
        return result.to(original_dtype)