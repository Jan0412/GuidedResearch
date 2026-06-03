import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    batch_id = tl.program_id(0)
    
    # Calculate base offsets for this batch
    base_offset = batch_id * seq_len
    
    # We'll process the sequence dimension in blocks
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load data
    x_ptrs = x_ptr + base_offset + offsets
    x = tl.load(x_ptrs, mask=mask, other=1.0)
    
    # Compute cumulative product
    # For the first block, just store the values
    # For subsequent blocks, multiply by the running product from previous blocks
    if block_start == 0:
        y = x
    else:
        # Get the cumulative product up to the start of this block
        prev_start = block_start - 1
        if prev_start >= 0:
            prev_ptr = y_ptr + base_offset + prev_start
            prev_val = tl.load(prev_ptr)
            y = x * prev_val
    
    # Store result
    y_ptrs = y_ptr + base_offset + offsets
    tl.store(y_ptrs, y, mask=mask)


# Optimized kernel for cumprod that handles multiple blocks per batch correctly
@triton.jit
def cumprod_kernel_optimized(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Process each batch in parallel
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * seq_len
    
    # Initialize running product
    running_prod = 1.0
    
    # Process in chunks
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values
        x_ptrs = x_ptr + base_offset + offsets
        x = tl.load(x_ptrs, mask=mask, other=1.0)
        
        # Update running product: multiply by previous running product for first element of block
        if block_idx > 0:
            # Load previous element's output to get cumulative product up to this point
            prev_idx = block_start - 1
            prev_ptr = y_ptr + base_offset + prev_idx
            prev_val = tl.load(prev_ptr)
            # Apply cumulative product for this block
            x = x * prev_val
        
        # Compute cumulative product within the block
        cumprod_vals = tl.cumprod(x, axis=0)
        
        # Store result
        y_ptrs = y_ptr + base_offset + offsets
        tl.store(y_ptrs, cumprod_vals, mask=mask)


# Even more optimized kernel that computes cumulative product in a single pass per batch
@triton.jit
def cumprod_kernel_fused(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each batch gets its own program
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * seq_len
    
    # Initialize running product
    running_prod = tl.load(x_ptr + base_offset)
    tl.store(y_ptr + base_offset, running_prod)
    
    # Process remaining elements
    for i in range(1, seq_len):
        offset = base_offset + i
        x_val = tl.load(x_ptr + offset)
        running_prod = running_prod * x_val
        tl.store(y_ptr + offset, running_prod)


# Better implementation using multiple blocks with prefix-scan approach
@triton.jit
def cumprod_kernel_blockwise(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each batch and block combination is handled by a program
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    base_offset = batch_id * seq_len
    block_start = block_id * BLOCK_SIZE
    
    # Calculate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load input values
    x_ptrs = x_ptr + base_offset + offsets
    x = tl.load(x_ptrs, mask=mask, other=1.0)
    
    # Compute cumulative product within block
    cumprod_block = tl.cumprod(x, axis=0)
    
    # Store intermediate result
    y_ptrs = y_ptr + base_offset + offsets
    tl.store(y_ptrs, cumprod_block, mask=mask)
    
    # For blocks beyond the first, we need to multiply by the cumulative product up to the end of previous block
    if block_id > 0:
        # Get the last element of the previous block's output
        prev_block_end = block_start - 1
        if prev_block_end >= 0:
            prev_ptr = y_ptr + base_offset + prev_block_end
            prev_val = tl.load(prev_ptr)
            
            # Multiply this block's cumulative product by the previous value
            tl.store(y_ptrs, cumprod_block * prev_val, mask=mask)


# Final optimized implementation using prefix-scan approach
@triton.jit
def cumprod_kernel_final(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each batch gets its own program
    batch_id = tl.program_id(0)
    base_offset = batch_id * seq_len
    
    # Process in blocks
    for block_idx in range(NUM_BLOCKS):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values
        x_ptrs = x_ptr + base_offset + offsets
        x = tl.load(x_ptrs, mask=mask, other=1.0)
        
        # If this is not the first block, multiply by the cumulative product up to the end of previous block
        if block_idx > 0:
            prev_block_end = block_start - 1
            prev_ptr = y_ptr + base_offset + prev_block_end
            prev_val = tl.load(prev_ptr)
            x = x * prev_val
        
        # Compute cumulative product within the current block
        cumprod_block = tl.cumprod(x, axis=0)
        
        # Store result
        y_ptrs = y_ptr + base_offset + offsets
        tl.store(y_ptrs, cumprod_block, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of cumulative product.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute cumulative product
        
    Returns:
        Tensor with cumulative product along specified dimension
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Get dimensions
    batch_size = 1
    if dim == 0:
        batch_size = x.shape[0]
        seq_len = 1
        for i in range(1, len(x.shape)):
            seq_len *= x.shape[i]
        x = x.view(batch_size, seq_len)
    else:
        # Handle general dimension by reshaping to 2D
        shape = x.shape
        batch_size = 1
        for i in range(dim):
            batch_size *= shape[i]
        seq_len = shape[dim]
        # Reshape: (batch_size, seq_len, remaining_dims) -> (batch_size*remaining_dims, seq_len)
        # Actually simpler: move dim to position 1, then reshape to 2D
        perm = list(range(len(shape)))
        perm[dim], perm[1] = perm[1], perm[dim]
        x = x.permute(perm)
        
        # Reshape to 2D: (batch_size, seq_len, *rest) -> (batch_size*rest, seq_len)
        if len(shape) > 2:
            rest = 1
            for i in range(2, len(shape)):
                rest *= shape[i]
            x = x.view(batch_size * rest, seq_len)
        else:
            x = x.view(batch_size, seq_len)
    
    output = torch.empty_like(x)
    
    # Calculate grid dimensions
    batch_size_out = x.shape[0]
    seq_len_out = x.shape[1]
    
    BLOCK_SIZE = 256
    NUM_BLOCKS = (seq_len_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use the final optimized kernel
    grid = lambda meta: (batch_size_out,)
    
    # Launch the kernel
    cumprod_kernel_final[grid](
        x,
        output,
        batch_size_out,
        seq_len_out,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BLOCKS=NUM_BLOCKS,
    )
    
    # Reshape back to original dimensions
    if dim != 1:
        # Need to reshape back
        shape = list(x.shape)
        shape[0] = batch_size
        shape[1] = seq_len
        if len(shape) > 2:
            rest = shape[2:]
            # First reshape to (batch_size, seq_len, *rest)
            temp = output.view(batch_size, seq_len, *rest)
            # Then permute back
            perm = list(range(len(shape)))
            perm[1], perm[dim] = perm[dim], perm[1]
            output = temp.permute(perm)
        else:
            output = output.view(batch_size, seq_len)
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product operation using Triton kernels.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using optimized Triton kernel for cumulative product.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying cumulative product along dim.
        """
        return triton_cumprod(x, dim=self.dim)