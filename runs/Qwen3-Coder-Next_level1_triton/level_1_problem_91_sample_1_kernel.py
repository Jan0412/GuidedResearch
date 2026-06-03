import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the outer dimension (non-dim dimension)
    batch_id = tl.program_id(0)
    
    # Calculate base pointer offset for this batch
    base_offset = batch_id * dim_stride
    
    # Process the dimension in reverse order
    # We'll use a block-based approach where each block processes a portion of the dimension
    
    # For each position in the dimension (in reverse)
    for start_idx in range(dim_size - 1, -1, -BLOCK_SIZE):
        idx = start_idx + tl.arange(0, BLOCK_SIZE)
        idx = idx[idx < dim_size]  # Mask out-of-bounds indices
        
        # Calculate actual offsets
        offsets = base_offset + idx * dim_stride
        
        # Load values
        mask = idx < dim_size
        vals = tl.load(x_ptr + offsets, mask=mask)
        
        # Perform reverse cumulative sum
        if start_idx == dim_size - 1:
            cumsum = vals
        else:
            # We need to accumulate from the previous iteration
            # Since we're going in reverse, we accumulate forward in the reversed order
            cumsum = vals + cumsum_prev
            
        cumsum_prev = cumsum
        
        # Store results
        tl.store(out_ptr + offsets, cumsum, mask=mask)


@triton.jit
def reverse_cumsum_optimized_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * dim_size
    
    # Create indices for this dimension
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < dim_size
    
    # Load all values at once
    x = tl.load(x_ptr + base_offset + idx, mask=mask)
    
    # Initialize reverse cumulative sum
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute reverse cumulative sum
    for i in range(dim_size - 1, -1, -1):
        cumsum = tl.where(idx == i, x[i] + cumsum, cumsum)
    
    # Store results
    tl.store(out_ptr + base_offset + idx, cumsum, mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumulative sum
        
    Returns:
        Tensor with reverse cumulative sum computed along dim
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and calculate strides
    shape = x.shape
    dim_size = shape[dim]
    batch_size = 1
    for i, s in enumerate(shape):
        if i != dim:
            batch_size *= s
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate stride for the dimension
    stride = x.stride(dim)
    
    # For 1D case or when dim=0, use simpler kernel
    if len(shape) == 1 or (dim == 0 and len(shape) == 1):
        BLOCK_SIZE = min(1024, dim_size)
        grid = (1,)
        
        @triton.jit
        def simple_reverse_cumsum_kernel(
            x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            # Process in reverse
            start = n_elements - 1
            cumsum = tl.zeros([1], dtype=tl.float32)
            
            for i in range(n_elements - 1, -1, -1):
                offset = i
                val = tl.load(x_ptr + offset)
                cumsum = cumsum + val
                tl.store(out_ptr + offset, cumsum)
        
        simple_reverse_cumsum_kernel[grid](
            x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Multi-dimensional case
        # Flatten all dimensions except the target dimension
        if dim != 0:
            # Move the target dimension to front for simpler processing
            x = x.transpose(0, dim).contiguous()
            out_transposed = torch.empty_like(x)
            
            # Calculate new parameters
            shape = x.shape
            dim_size = shape[0]
            batch_size = 1
            for s in shape[1:]:
                batch_size *= s
                
            BLOCK_SIZE = min(256, dim_size)
            grid = (batch_size,)
            
            @triton.jit
            def batched_reverse_cumsum_kernel(
                x_ptr, out_ptr, dim_size, batch_idx, BLOCK_SIZE: tl.constexpr
            ):
                # Calculate base offset for this batch
                # We need to compute offsets based on batch_idx and remaining dimensions
                # This is simplified - in practice would need proper index calculation
                
                # For simplicity, use a direct approach
                idx = tl.arange(0, BLOCK_SIZE)
                
                # Process each position in reverse
                cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                
                for i in range(dim_size - 1, -1, -1):
                    offset = i * batch_size + batch_idx
                    val = tl.load(x_ptr + offset)
                    cumsum = tl.where(idx == 0, val + cumsum, cumsum)
                
                # Store back
                for i in range(dim_size):
                    offset = i * batch_size + batch_idx
                    tl.store(out_ptr + offset, cumsum[i] if i < BLOCK_SIZE else 0)
            
            # For simplicity, use a more straightforward implementation
            BLOCK_SIZE = min(128, dim_size)
            grid = (batch_size,)
            
            def create_reverse_cumsum_kernel():
                @triton.jit
                def kernel(
                    x_ptr, out_ptr, dim_size, batch_size, batch_id, BLOCK_SIZE: tl.constexpr
                ):
                    # Calculate offsets for this batch
                    # Flatten remaining dimensions into a single index
                    temp_batch_id = batch_id
                    indices = []
                    remaining_dims = list(x.shape[1:])
                    
                    for d in reversed(remaining_dims):
                        indices.append(temp_batch_id % d)
                        temp_batch_id //= d
                    indices = list(reversed(indices))
                    
                    # Process the dimension in reverse
                    cumsum = 0.0
                    for i in range(dim_size - 1, -1, -1):
                        # Calculate offset for position i in this batch
                        offset = i
                        for j, idx in enumerate(indices):
                            offset = offset * x.shape[j + 1] + idx
                        offset = offset * x.stride(0) + i * x.stride(0)
                        
                        val = tl.load(x_ptr + offset)
                        cumsum += val
                        tl.store(out_ptr + offset, cumsum)
                
                return kernel
            
            # Use simpler approach for now
            BLOCK_SIZE = min(128, dim_size)
            grid = (batch_size,)
            
            @triton.jit
            def simple_batched_kernel(
                x_ptr, out_ptr, dim_size, batch_size, batch_id, 
                x_strides, out_strides, BLOCK_SIZE: tl.constexpr
            ):
                # Convert linear batch_id to multi-dimensional indices
                idx = batch_id
                dims = []
                for i in range(1, len(x.shape)):
                    dims.append(idx % x.shape[i])
                    idx //= x.shape[i]
                
                # Compute reverse cumsum for this batch
                cumsum = 0.0
                for i in range(dim_size - 1, -1, -1):
                    # Calculate offset for this position
                    offset = i
                    for j in range(1, len(x.shape)):
                        offset = offset * x.shape[j] + dims[j-1]
                    offset = offset * x.stride(0) + i * x.stride(0)
                    
                    val = tl.load(x_ptr + offset)
                    cumsum += val
                    tl.store(out_ptr + offset, cumsum)
            
            # Get strides
            x_strides = tuple(x.stride())
            out_strides = tuple(out_transposed.stride())
            
            # Launch kernel
            simple_batched_kernel[grid](
                x, out_transposed, dim_size, batch_size, 
                tl.arange(0, 1),  # placeholder
                x_strides, out_strides,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Transpose back
            out = out_transposed.transpose(0, dim).contiguous()
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs reverse cumulative sum using Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)