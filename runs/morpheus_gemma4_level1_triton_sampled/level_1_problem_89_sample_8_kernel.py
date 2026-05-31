import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def local_scan_kernel(
    x_ptr, 
    out_ptr, 
    block_sums_ptr,
    stride_row, 
    stride_bs_row,
    seq_len, 
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    # Calculate offsets for the current block in the row
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load input values
    x = tl.load(x_ptr + pid_row * stride_row + offsets, mask=mask, other=0.0)
    
    # Compute local cumulative sum
    # tl.cumsum is available in Triton 2.0+
    res = tl.cumsum(x, axis=0)
    
    # Store the local cumsum results
    tl.store(out_ptr + pid_row * stride_row + offsets, res, mask=mask)
    
    # Store the total sum of this block for the second pass
    # The total sum is the sum of all elements in the block
    block_total = tl.sum(x, axis=0)
    tl.store(block_sums_ptr + pid_row * stride_bs_row + pid_block, block_total)

@triton.jit
def block_scan_kernel(
    block_sums_ptr,
    stride_bs_row,
    num_blocks,
    BLOCK_SIZE_BS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    
    # Each program handles one row's block totals
    offsets = tl.arange(0, BLOCK_SIZE_BS)
    mask = offsets < num_blocks
    
    # Load block totals
    sums = tl.load(block_sums_ptr + pid_row * stride_bs_row + offsets, mask=mask, other=0.0)
    
    # Compute prefix sum of block totals
    res = tl.cumsum(sums, axis=0)
    
    # Store results back
    tl.store(block_sums_ptr + pid_row * stride_bs_row + offsets, res, mask=mask)

@triton.jit
def final_add_kernel(
    out_ptr, 
    block_sums_ptr,
    stride_row, 
    stride_bs_row,
    seq_len, 
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    # The first block does not need an offset added
    if pid_block == 0:
        return
    
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load the prefix sum of all previous blocks
    # block_sums[pid_row, pid_block - 1] contains the sum of all blocks from 0 to pid_block-1
    prev_sum = tl.load(block_sums_ptr + pid_row * stride_bs_row + pid_block - 1)
    
    # Load existing local cumsums
    val = tl.load(out_ptr + pid_row * stride_row + offsets, mask=mask, other=0.0)
    
    # Add the offset and store
    tl.store(out_ptr + pid_row * stride_row + offsets, val + prev_sum, mask=mask)

def triton_cumsum(x: torch.Tensor, dim: int):
    # Move the target dimension to the end and reshape to 2D (batch, seq_len)
    original_shape = x.shape
    if dim < 0:
        dim = x.dim() + dim
        
    # Transpose target dim to the last position
    x = x.transpose(dim, -1)
    # Flatten all dimensions except the last one
    seq_len = x.shape[-1]
    x = x.reshape(-1, seq_len).contiguous()
    batch_size = x.shape[0]
    
    # Triton parameters
    BLOCK_SIZE = 1024
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Find the smallest power of 2 >= num_blocks for the block_scan_kernel
    BLOCK_SIZE_BS = 1 << (num_blocks - 1).bit_length()
    
    # Output and helper tensors
    out = torch.empty_like(x)
    block_sums = torch.empty((batch_size, num_blocks), device=x.device, dtype=x.dtype)
    
    stride_row = x.stride(1) # This is actually seq_len for contiguous x
    stride_bs_row = num_blocks
    
    # Pass 1: Local scan
    grid_local = (batch_size, num_blocks)
    local_scan_kernel[grid_local](
        x, out, block_sums, 
        seq_len, stride_row, stride_bs_row, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Pass 2: Scan the block totals
    grid_block = (batch_size,)
    block_scan_kernel[grid_block](
        block_sums, stride_bs_row, num_blocks, 
        BLOCK_SIZE_BS=BLOCK_SIZE_BS
    )
    
    # Pass 3: Final addition of offsets
    grid_final = (batch_size, num_blocks)
    final_add_kernel[grid_final](
        out, block_sums, 
        seq_len, stride_row, stride_bs_row, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape and transpose back to original layout
    out = out.view(*original_shape[:dim], *original_shape[dim+1:], seq_len)
    # Move the last dimension back to the original 'dim' position
    # We need to identify where the last dim was moved from
    dims = list(range(out.dim()))
    # The last dim (out.dim()-1) should move to 'dim'
    # The dims from 'dim' to 'out.dim()-2' shift right
    perm = dims[:-1]
    perm.insert(dim, perm.pop())
    out = out.permute(*perm).contiguous()
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs a cumulative sum (prefix sum) operation along a specified dimension,
    optimized with custom Triton kernels.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Ensure input is on GPU and FP32
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)
        
        # For very small sequence lengths, torch.cumsum is already very fast
        # But we use our Triton implementation for the target speedup.
        return triton_cumsum(x, self.dim)