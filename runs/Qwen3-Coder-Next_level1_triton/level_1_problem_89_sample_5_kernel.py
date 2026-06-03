import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dimension dim
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between elements along sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Compute base pointer for this batch
    batch_ptr = x_ptr + batch_id * stride_batch
    
    # Create offsets for sequence elements
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in chunks of BLOCK_SIZE
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute current offsets
        current_offsets = start + offsets
        mask = current_offsets < seq_len
        
        # Load current values
        x = tl.load(batch_ptr + current_offsets * stride_seq, mask=mask, other=0.0)
        
        # Compute cumulative sum
        # For efficiency, we use a simple inclusive scan approach
        # Each block processes sequentially within itself
        # Then we accumulate across blocks
        
        # First pass: compute running sum within the block
        running_sum = tl.zeros_like(x)
        for i in range(tl.where(mask, 1, 0).sum()):
            if i == 0:
                running_sum = tl.where(mask, x, 0.0)
            else:
                prev_offset = current_offsets - (i - tl.arange(0, BLOCK_SIZE) % BLOCK_SIZE)
                # We need to be careful with the indices - simpler approach:
                # We'll do a simple in-block cumulative sum
                pass
        
        # Simpler approach: sequential cumsum within block
        # This is more straightforward and works well for reasonable BLOCK_SIZE
        for i in range(BLOCK_SIZE):
            if i == 0:
                # First element is just x[0] if within bounds, else 0
                elem = tl.load(batch_ptr + (start + i) * stride_seq, mask=(start + i) < seq_len, other=0.0)
                tl.store(out_ptr + batch_id * stride_batch + (start + i) * stride_seq, elem, mask=(start + i) < seq_len)
            else:
                prev = tl.load(out_ptr + batch_id * stride_batch + (start + i - 1) * stride_seq, mask=(start + i - 1) < seq_len, other=0.0)
                curr = tl.load(batch_ptr + (start + i) * stride_seq, mask=(start + i) < seq_len, other=0.0)
                cumsum_val = prev + curr
                tl.store(out_ptr + batch_id * stride_batch + (start + i) * stride_seq, cumsum_val, mask=(start + i) < seq_len)


# Optimized version using parallel scan algorithm for better performance
@triton.jit
def cumsum_optimized_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dimension dim
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between elements along sequence dimension
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Shared memory for scan operations
    # We'll use a two-pass algorithm: up-sweep (reduce) and down-sweep (scan)
    
    # Each block handles one batch
    batch_id = tl.program_id(0)
    
    # Compute base pointer for this batch
    batch_ptr = x_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Shared memory buffers
    # We'll use a simpler approach for FP32: process sequentially within blocks
    # For very long sequences, we can do parallel scan, but for simplicity and robustness:
    
    # First pass: copy input to output and compute local cumulative sums
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(0, seq_len, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < seq_len
        
        # Load data
        x = tl.load(batch_ptr + current_offsets * stride_seq, mask=mask, other=0.0)
        
        # Compute cumulative sum within this chunk
        cumsum = tl.zeros_like(x)
        for i in range(BLOCK_SIZE):
            if i == 0:
                cumsum = tl.where(mask, x, 0.0)
            else:
                prev = tl.load(out_batch_ptr + current_offsets * stride_seq - stride_seq, mask=(current_offsets > 0), other=0.0)
                cumsum = tl.where(mask, prev + x, 0.0)
                break  # We only need the last value for the next block
        
        # Store intermediate result
        tl.store(out_batch_ptr + current_offsets * stride_seq, cumsum, mask=mask)


# Even simpler and more reliable approach for cumsum
@triton.jit
def cumsum_simple_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dimension dim
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between elements along sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Compute base pointer for this batch
    batch_ptr = x_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Sequential cumsum along the sequence dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(0, seq_len, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < seq_len
        
        # Load current input values
        x = tl.load(batch_ptr + current_offsets * stride_seq, mask=mask, other=0.0)
        
        # Compute cumulative sum
        # For the first block, it's just cumulative sum of x
        # For subsequent blocks, we need to add the sum of all previous blocks
        
        # First, do local cumulative sum
        cumsum = tl.zeros_like(x)
        for i in range(BLOCK_SIZE):
            if i == 0:
                cumsum = tl.where(mask, x, 0.0)
            else:
                # Get previous element from output (already computed)
                prev = tl.load(out_batch_ptr + (current_offsets - 1) * stride_seq, mask=(current_offsets > 0), other=0.0)
                cumsum = tl.where(mask, prev + x, 0.0)
        
        # Store results
        tl.store(out_batch_ptr + current_offsets * stride_seq, cumsum, mask=mask)


# Efficient parallel implementation using a two-pass algorithm
@triton.jit
def cumsum_parallel_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dimension dim
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between elements along sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # This implements a simple parallel scan algorithm
    # For simplicity and robustness, we'll use an optimized sequential approach
    # that benefits from GPU parallelism across batches
    
    batch_id = tl.program_id(0)
    
    # Compute base pointers
    batch_ptr = x_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Sequential processing within each batch (but batches are parallel)
    # This is actually optimal for cumsum since it's inherently sequential
    # But we can process each batch in parallel
    
    # Handle one element per thread for better parallelism
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Since cumsum is sequential, we need to do it sequentially per batch
    # But we can parallelize across batches
    
    # Actually, let's do a simple loop-based approach per batch
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially through the sequence
    for start in range(0, seq_len, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < seq_len
        
        # Load input
        x = tl.load(batch_ptr + current_offsets * stride_seq, mask=mask, other=0.0)
        
        # Compute cumulative sum
        if start == 0:
            cumsum = x
            for i in range(1, BLOCK_SIZE):
                cumsum = tl.where(mask & (current_offsets >= i), cumsum + tl.load(batch_ptr + (current_offsets - i) * stride_seq, mask=mask & (current_offsets >= i), other=0.0), cumsum)
        else:
            # Load previous cumsum value
            prev_cumsum = tl.load(out_batch_ptr + (start - 1) * stride_seq, mask=(start - 1) < seq_len, other=0.0)
            cumsum = x + prev_cumsum
            for i in range(1, BLOCK_SIZE):
                offset = start + i
                if offset < seq_len:
                    cumsum = tl.where(mask & (current_offsets >= offset), cumsum + tl.load(batch_ptr + (offset - 1) * stride_seq, mask=mask & (current_offsets >= offset), other=0.0), cumsum)
        
        tl.store(out_batch_ptr + current_offsets * stride_seq, cumsum, mask=mask)


# Final optimized version - simple but effective
@triton.jit
def cumsum_final_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dimension dim
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between elements along sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each batch is processed in parallel
    batch_id = tl.program_id(0)
    
    # Compute base pointers
    batch_ptr = x_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Sequential cumsum - we process sequentially because cumsum is inherently sequential
    # But we parallelize across batches which is the main benefit
    for i in range(seq_len):
        # Load current input value
        x_val = tl.load(batch_ptr + i * stride_seq, mask=i < seq_len, other=0.0)
        
        # Get previous cumsum value
        if i == 0:
            cumsum_val = x_val
        else:
            prev_cumsum = tl.load(out_batch_ptr + (i - 1) * stride_seq, mask=i > 0, other=0.0)
            cumsum_val = prev_cumsum + x_val
        
        # Store result
        tl.store(out_batch_ptr + i * stride_seq, cumsum_val, mask=i < seq_len)


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Triton-based cumulative sum implementation.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute cumulative sum
        
    Returns:
        torch.Tensor: Output tensor with cumulative sum
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Compute strides
    stride = x.stride()
    
    # Calculate batch size and sequence length
    # We flatten everything except the target dimension
    batch_size = 1
    for i in range(dim):
        batch_size *= shape[i]
    seq_len = shape[dim]
    
    # For non-contiguous tensors, we need to handle strides properly
    stride_batch = 1
    for i in range(dim):
        stride_batch *= shape[i] if i == 0 else shape[i] // stride[i-1] if i > 0 else 1
    
    # Calculate actual strides for our kernel
    # stride_batch: distance between start of consecutive batches
    # stride_seq: distance between consecutive elements along dim
    
    stride_batch = 1
    for i in range(dim):
        stride_batch *= shape[i]
    
    # For the target dimension, the stride should be 1 for contiguous tensors
    # But let's compute it properly
    stride_seq = stride[dim]
    
    # For 2D case where dim=1 and tensor is contiguous
    if len(shape) == 2 and dim == 1 and x.is_contiguous():
        batch_size = shape[0]
        seq_len = shape[1]
        stride_batch = shape[1]  # Distance between starts of batches
        stride_seq = 1           # Distance between elements in sequence
    else:
        # General case - compute strides properly
        # We'll use a simpler approach: permute to move dim to last position
        # This makes implementation much simpler
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        
        # Transpose tensor
        x_perm = x.permute(perm)
        out_perm = torch.empty_like(x_perm)
        
        # Now dim is the last dimension
        new_shape = x_perm.shape
        stride_perm = x_perm.stride()
        
        # Compute batch size (all dimensions except last)
        batch_size = 1
        for i in range(len(new_shape) - 1):
            batch_size *= new_shape[i]
        seq_len = new_shape[-1]
        
        stride_batch = stride_perm[-1]  # Usually 1 for contiguous
        stride_seq = 1
        
        # For contiguous tensor, stride_batch should be seq_len
        if x_perm.is_contiguous():
            stride_batch = seq_len
            stride_seq = 1
        
        # Launch kernel for transposed tensor
        BLOCK_SIZE = min(1024, seq_len)
        grid = (batch_size,)
        
        cumsum_final_kernel[grid](
            x_perm.contiguous(), 
            out_perm, 
            batch_size, 
            seq_len, 
            stride_batch, 
            stride_seq,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Invert permutation
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        return out_perm.permute(inv_perm)
    
    # Special case for 2D contiguous tensors
    BLOCK_SIZE = min(1024, seq_len)
    grid = (batch_size,)
    
    cumsum_final_kernel[grid](
        x, 
        out, 
        batch_size, 
        seq_len, 
        stride_batch, 
        stride_seq,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Scan model using Triton kernels for cumulative sum operation.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the optimized Scan model, computing the cumulative sum along the specified dimension using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return triton_cumsum(x, self.dim)