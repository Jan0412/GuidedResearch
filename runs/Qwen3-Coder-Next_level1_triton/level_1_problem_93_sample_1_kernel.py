import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length (size along the dimension)
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between sequence elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes masked cumulative sum along the last dimension.
    Uses parallel prefix sum algorithm.
    """
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_batch_ptr = x_ptr + batch_id * stride_batch
    mask_batch_ptr = mask_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Shared memory for the prefix sum computation
    # We use a slightly larger block to handle power-of-2 requirements
    BLOCK_SIZE_POW2 = 1
    while BLOCK_SIZE_POW2 < BLOCK_SIZE:
        BLOCK_SIZE_POW2 *= 2
    
    # Allocate shared memory
    # Note: Triton requires compile-time constants for shared memory allocation
    # So we'll use a workaround with dynamic shared memory
    
    # For simplicity and compatibility, we'll use a fixed maximum block size
    MAX_BLOCK_SIZE = 2048  # Sufficient for most use cases
    tl.static_assert(BLOCK_SIZE <= MAX_BLOCK_SIZE, "BLOCK_SIZE must be <= MAX_BLOCK_SIZE")
    
    # Load data into registers and compute prefix sum
    # We'll implement a simple sequential approach for sequences <= BLOCK_SIZE
    # For longer sequences, we'd need a more complex parallel algorithm
    
    # For this implementation, we'll use a straightforward approach that works
    # for sequences up to BLOCK_SIZE elements per batch
    # For longer sequences, we'd need a hierarchical approach
    
    # Since the problem size can be large (32768), we'll use a scan-based approach
    # with multiple passes
    
    # First pass: compute partial sums in blocks
    # We'll process the sequence in chunks of size BLOCK_SIZE
    
    # Initialize running sum
    running_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process sequence in chunks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute end of current chunk
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        count = end - start
        
        # Load mask and x values
        offsets = tl.arange(0, BLOCK_SIZE)
        mask_offsets = offsets < count
        
        # Load x values
        x_offsets = start + offsets
        x_val = tl.load(x_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0.0)
        
        # Load mask values
        mask_val = tl.load(mask_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0)
        mask_val = mask_val.to(tl.float32)
        
        # Apply mask and compute running sum
        masked_x = x_val * mask_val
        
        # Sequential prefix sum within the block
        for i in range(BLOCK_SIZE):
            if i < count:
                running_sum = running_sum + masked_x[i]
                # Store the cumulative sum
                if mask_offsets[i]:
                    tl.store(out_batch_ptr + (start + i) * stride_seq, running_sum)
                else:
                    tl.store(out_batch_ptr + (start + i) * stride_seq, running_sum)


@triton.jit
def masked_cumsum_kernel_optimized(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length (size along the dimension)
    stride_batch,  # Stride between batches
    stride_seq,  # Stride between sequence elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized masked cumulative sum using a two-pass algorithm:
    1. Compute block sums
    2. Propagate sums and compute final values
    """
    # Process one batch per program
    batch_id = tl.program_id(0)
    
    # Calculate base pointers
    x_batch_ptr = x_ptr + batch_id * stride_batch
    mask_batch_ptr = mask_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Allocate shared memory for block sums
    # Maximum 16 blocks for seq_len up to 32768 with BLOCK_SIZE=2048
    MAX_BLOCKS = 32
    block_sums = tl.zeros((MAX_BLOCKS,), dtype=tl.float32)
    
    # First pass: compute block sums
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        count = end - start
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask_offsets = offsets < count
        
        # Load x values
        x_offsets = start + offsets
        x_val = tl.load(x_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0.0)
        
        # Load mask values
        mask_val = tl.load(mask_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0)
        mask_val = mask_val.to(tl.float32)
        
        # Apply mask and compute block sum
        masked_x = x_val * mask_val
        block_sum = tl.sum(masked_x, axis=0)
        block_sums[block_idx] = block_sum
    
    # Second pass: compute final values with prefix sums
    prefix_sum = tl.zeros((1,), dtype=tl.float32)
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        count = end - start
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask_offsets = offsets < count
        
        # Load x values
        x_offsets = start + offsets
        x_val = tl.load(x_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0.0)
        
        # Load mask values
        mask_val = tl.load(mask_batch_ptr + x_offsets * stride_seq, mask=mask_offsets, other=0)
        mask_val = mask_val.to(tl.float32)
        
        # Apply mask and compute running sum
        masked_x = x_val * mask_val
        
        # Sequential prefix sum within the block
        for i in range(BLOCK_SIZE):
            if i < count:
                prefix_sum = prefix_sum + masked_x[i]
                # Store the cumulative sum
                if mask_offsets[i]:
                    tl.store(out_batch_ptr + (start + i) * stride_seq, prefix_sum)
                else:
                    tl.store(out_batch_ptr + (start + i) * stride_seq, prefix_sum)
    
    # For this implementation, we'll use a simpler approach that works well
    # for the given problem size


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Computes masked cumulative sum along specified dimension.
    
    Args:
        x: Input tensor
        mask: Boolean mask tensor
        dim: Dimension along which to compute cumulative sum
    
    Returns:
        torch.Tensor: Masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "Input and mask must have the same shape."
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Convert mask to float for computation
    mask_float = mask.to(torch.float32)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    batch_size = 1
    seq_len = x.shape[dim]
    
    # Calculate batch dimension (all dimensions except the target dim)
    for i, s in enumerate(x.shape):
        if i != dim:
            batch_size *= s
    
    # Calculate strides
    stride_batch = x.stride(dim) if dim == 0 else x.stride(dim) * x.shape[dim]
    
    # Actually, let's compute proper strides
    # We need to handle arbitrary dimensions properly
    # Reshape to 2D: [batch, seq_len] where seq_len is along dim
    original_shape = x.shape
    
    # Move the target dimension to the end
    if dim != len(x.shape) - 1:
        x = x.movedim(dim, -1)
        mask_float = mask_float.movedim(dim, -1)
        out = out.movedim(dim, -1)
    
    # Reshape to 2D: [batch_size, seq_len]
    x_flat = x.reshape(-1, x.shape[-1])
    mask_flat = mask_float.reshape(-1, x.shape[-1])
    out_flat = out.reshape(-1, x.shape[-1])
    
    batch_size = x_flat.shape[0]
    seq_len = x_flat.shape[1]
    
    # Set block size
    BLOCK_SIZE = 128
    
    # Grid: one block per batch
    grid = lambda meta: (batch_size,)
    
    # Launch kernel - but we need to implement the actual algorithm properly
    # For now, let's use a simpler approach with a loop
    # Actually, let me implement a proper kernel that works
    
    # Since the optimized kernel above has complexity issues, let me use
    # a simpler approach that works for the given problem size
    
    # For this specific problem (dim=1, large seq_len), let's use:
    # - 1D grid: one program per batch
    # - Sequential processing within each batch
    
    # However, for better performance, I'll implement a more efficient version
    # that processes in parallel
    
    # Use a kernel that processes the cumulative sum efficiently
    from triton import ops
    
    # Actually, let me implement a proper masked cumulative sum kernel
    # that handles the parallel prefix sum correctly
    
    # For the given problem size (32768), we can use a simple approach
    # but for optimal performance, let's use a block-wise algorithm
    
    # Since the kernel implementation is complex, let me use a simpler
    # but still optimized approach
    
    # First, create a kernel that processes each batch sequentially
    # but does the cumsum efficiently
    
    # For this implementation, I'll use a straightforward approach that
    # works well for the given problem size
    
    # Let me implement a proper kernel
    pass


# Actually, let me implement a correct and efficient kernel
@triton.jit
def masked_cumsum_final_kernel(
    x_ptr, 
    mask_ptr, 
    out_ptr,
    n_batches,
    seq_len,
    stride_batch,
    stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Final implementation of masked cumulative sum kernel.
    Uses a simple but efficient approach for the given problem size.
    """
    batch_id = tl.program_id(0)
    
    # Calculate base pointers
    x_batch = x_ptr + batch_id * stride_batch
    mask_batch = mask_ptr + batch_id * stride_batch
    out_batch = out_ptr + batch_id * stride_batch
    
    # Running sum
    cumsum = tl.zeros((1,), dtype=tl.float32)
    
    # Process sequence elements sequentially
    for i in range(seq_len):
        # Calculate offset
        offset = i * stride_seq
        
        # Load values
        x_val = tl.load(x_batch + offset)
        mask_val = tl.load(mask_batch + offset)
        
        # Apply mask
        if mask_val > 0:
            cumsum = cumsum + x_val
        
        # Store result
        tl.store(out_batch + offset, cumsum)


class ModelNew(nn.Module):
    """
    Optimized version of Model with custom Triton kernels.
    Computes masked cumulative sum along specified dimension.
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
        # Ensure inputs are on CUDA
        assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Convert mask to float for computation
        mask_float = mask.to(torch.float32)
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        original_shape = x.shape
        dim = self.dim
        
        # Move target dimension to the end for easier processing
        if dim != len(x.shape) - 1:
            x = x.movedim(dim, -1)
            mask_float = mask_float.movedim(dim, -1)
            out = out.movedim(dim, -1)
        
        # Reshape to 2D
        x_flat = x.reshape(-1, x.shape[-1])
        mask_flat = mask_float.reshape(-1, x.shape[-1])
        out_flat = out.reshape(-1, x.shape[-1])
        
        batch_size = x_flat.shape[0]
        seq_len = x_flat.shape[1]
        
        # Calculate strides
        stride_batch = x_flat.stride(0)
        stride_seq = x_flat.stride(1)
        
        # Set block size (not used in this simple implementation, but kept for API consistency)
        BLOCK_SIZE = 128
        
        # Grid: one program per batch
        grid = (batch_size,)
        
        # Launch kernel
        masked_cumsum_final_kernel[grid](
            x_flat, mask_flat, out_flat,
            batch_size, seq_len, stride_batch, stride_seq,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape back to original shape
        out = out_flat.reshape(out.shape)
        
        # Move dimension back if needed
        if dim != len(original_shape) - 1:
            # Calculate the new dimension position
            new_dim = dim
            if dim < len(original_shape) - 1:
                new_dim = dim
            out = out.movedim(-1, dim)
        
        return out