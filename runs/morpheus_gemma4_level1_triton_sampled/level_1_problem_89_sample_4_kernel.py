import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def local_scan_kernel(
    x_ptr, 
    out_ptr, 
    sums_ptr, 
    stride_xb, stride_xn, 
    stride_ob, stride_on, 
    stride_sb, stride_sn, 
    N, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for row and block
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    
    # Calculate offsets for the current block
    offsets = col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load the input block
    x = tl.load(x_ptr + row_id * stride_xb + offsets * stride_xn, mask=mask, other=0.0)
    
    # Compute local cumulative sum within the block
    res = tl.cumsum(x, axis=0)
    
    # Store the local cumulative sum results
    tl.store(out_ptr + row_id * stride_ob + offsets * stride_on, res, mask=mask)
    
    # Store the total sum of this block (the last element of the local cumsum)
    # We use tl.sum to be explicit and handle masking correctly
    block_sum = tl.sum(x, axis=0)
    tl.store(sums_ptr + row_id * stride_sb + col_block_id * stride_sn, block_sum)

@triton.jit
def sums_scan_kernel(
    sums_ptr, 
    stride_sb, stride_sn, 
    num_blocks, 
    BLOCK_SIZE_SUMS: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE_SUMS)
    mask = offsets < num_blocks
    
    # Load the block sums for the row
    s = tl.load(sums_ptr + row_id * stride_sb + offsets * stride_sn, mask=mask, other=0.0)
    
    # Compute the cumulative sum of the block sums
    res = tl.cumsum(s, axis=0)
    
    # Store the updated block sums (now prefix sums of blocks)
    tl.store(sums_ptr + row_id * stride_sb + offsets * stride_sn, res, mask=mask)

@triton.jit
def apply_offsets_kernel(
    out_ptr, 
    sums_ptr, 
    stride_ob, stride_on, 
    stride_sb, stride_sn, 
    N, 
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    
    # The first block (col_block_id == 0) needs no offset
    if col_block_id == 0:
        return
    
    offsets = col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load the offset from the previous block's prefix sum
    offset = tl.load(sums_ptr + row_id * stride_sb + (col_block_id - 1) * stride_sn)
    
    # Load local cumsum, add offset, and store back
    val = tl.load(out_ptr + row_id * stride_ob + offsets * stride_on, mask=mask, other=0.0)
    tl.store(out_ptr + row_id * stride_ob + offsets * stride_on, val + offset, mask=mask)

def triton_cumsum(x: torch.Tensor, dim: int):
    # Handle dimension: we optimize for the last dimension (dim=1 for 2D)
    # If dim is not the last dimension, we transpose, compute, and transpose back
    original_shape = x.shape
    ndim = x.ndim
    
    # Normalize dim to positive
    if dim < 0:
        dim += ndim
        
    if dim != ndim - 1:
        # Move target dimension to the end
        perm = list(range(ndim))
        perm.pop(dim)
        perm.append(dim)
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        x = x.permute(*perm).contiguous()
        res = triton_cumsum(x, ndim - 1)
        return res.permute(*inv_perm).contiguous()

    # Ensure input is contiguous and FP32
    x = x.contiguous().float()
    B = x.shape[0]
    N = x.shape[-1]
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Parameters for tiling
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Temporary tensor to store the sum of each block
    sums = torch.empty((B, num_blocks), device=x.device, dtype=torch.float32)
    
    # Strides
    stride_xb, stride_xn = x.stride()
    stride_ob, stride_on = out.stride()
    stride_sb, stride_sn = sums.stride()
    
    # 1. Local scan and block sum calculation
    local_grid = (B, num_blocks)
    local_scan_kernel[local_grid](
        x, out, sums, 
        stride_xb, stride_xn, 
        stride_ob, stride_on, 
        stride_sb, stride_sn, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 2. Prefix sum of the block sums
    # We use a block size large enough to cover all blocks per row (32768/1024=32)
    sums_grid = (B,)
    sums_scan_kernel[sums_grid](
        sums, 
        stride_sb, stride_sn, 
        num_blocks, 
        BLOCK_SIZE_SUMS=128 
    )
    
    # 3. Apply offsets from block sums to the local scans
    apply_grid = (B, num_blocks)
    apply_offsets_kernel[apply_grid](
        out, sums, 
        stride_ob, stride_on, 
        stride_sb, stride_sn, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a cumulative sum operation along a specified dimension
    using custom Triton kernels.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Perform cumsum using Triton kernels
        return triton_cumsum(x, self.dim)