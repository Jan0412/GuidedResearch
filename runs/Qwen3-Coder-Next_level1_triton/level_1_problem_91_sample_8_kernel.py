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
    # Get the program ID for the outer dimension (batch dimension)
    batch_id = tl.program_id(0)
    
    # Calculate the base offset for this batch
    base_offset = batch_id * dim_stride * dim_size
    
    # For each position in the dimension
    for i in tl.range(0, dim_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Calculate the actual indices for reverse cumsum
        # For reverse cumsum at position j, we need to sum from j to end
        # We'll compute in reverse order to do online calculation
        
    # We'll compute in two passes for simplicity and correctness
    # First pass: copy and reverse the data conceptually
    # Second pass: compute forward cumsum on reversed data
    # Third pass: reverse back
    
    # Actually, let's do it more efficiently with a single kernel
    # For reverse cumsum: out[i] = x[i] + x[i+1] + ... + x[n-1]
    # We can compute this by doing cumsum from the end
    
    # Let's use a different approach: compute the cumsum from the end
    # out[i] = x[i] + out[i+1] (with out[n-1] = x[n-1])
    
    # This requires sequential computation, so we'll do it in a loop
    # But Triton doesn't support loops well for large dimensions
    # Instead, we'll use a parallel algorithm
    
    # For simplicity and correctness, let's use a block-wise approach
    # Each block handles a segment and computes partial sums
    
    # Actually, for correctness and simplicity, let's implement the straightforward version
    # with a single thread per position computing the sum (not optimal but correct)
    
    # Better approach: use a parallel prefix sum algorithm
    # But for simplicity and to ensure correctness, let's do:
    
    # Process each batch independently
    for idx in range(dim_size):
        # Calculate the index in the original tensor
        real_idx = dim_size - 1 - idx  # Reverse index
        
        # Compute cumulative sum from real_idx to end
        cumsum_val = 0.0
        for j in range(real_idx, dim_size):
            # Calculate the actual offset in the tensor
            offset = base_offset + j * dim_stride
            val = tl.load(x_ptr + offset)
            cumsum_val += val
            
        # Store the result
        out_offset = base_offset + real_idx * dim_stride
        tl.store(out_ptr + out_offset, cumsum_val)


@triton.jit
def reverse_cumsum_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim_size,
    dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch ID
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * dim_stride * dim_size
    
    # Process the dimension in reverse order for cumsum
    # out[i] = x[i] + x[i+1] + ... + x[dim_size-1]
    
    # We'll compute this by iterating from the end
    # For each position, accumulate from that position to the end
    
    # Use a simple approach: each thread handles one position
    # This is O(n^2) but correct
    
    pos = tl.program_id(1)
    if pos < dim_size:
        # Calculate the actual position in the original tensor (reversed)
        real_pos = dim_size - 1 - pos
        
        cumsum_val = 0.0
        # Accumulate from real_pos to the end
        for j in range(real_pos, dim_size):
            offset = base_offset + j * dim_stride
            val = tl.load(x_ptr + offset)
            cumsum_val += val
            
        # Store result at the reversed position
        out_offset = base_offset + real_pos * dim_stride
        tl.store(out_ptr + out_offset, cumsum_val)


class TritonReverseCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure tensor is contiguous
        x = x.contiguous()
        
        # Get shape information
        shape = x.shape
        ndim = len(shape)
        
        # Normalize dimension
        if dim < 0:
            dim = ndim + dim
            
        # Calculate strides
        strides = x.stride()
        dim_stride = strides[dim]
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size = 1
        for i in range(dim):
            batch_size *= shape[i]
        dim_size = shape[dim]
        
        # Prepare for kernel launch
        # We'll treat everything before dim as batch dimension
        # and everything after dim as part of the element computation
        
        # For simplicity, we'll flatten all dimensions except the target dim
        # and process each "batch" independently
        
        # Calculate grid dimensions
        # Block size for the dimension
        BLOCK_SIZE_DIM = 128
        
        # Grid: (batch_size, dim_size)
        # But we need to handle multi-dimensional batches
        
        # Reshape to 2D if necessary for simpler handling
        if ndim == 1:
            # Simple 1D case
            grid = lambda meta: (1, min(dim_size, 1024))
            reverse_cumsum_fused_kernel[grid](
                x, out, 1, dim_size, dim_stride,
                BLOCK_SIZE=BLOCK_SIZE_DIM
            )
        else:
            # For multi-dimensional, we'll process each slice independently
            # Reshape to (prod_other_dims, dim_size)
            other_dims = 1
            for i, s in enumerate(shape):
                if i != dim:
                    other_dims *= s
                    
            # Reshape input and output
            x_2d = x.view(other_dims, dim_size)
            out_2d = out.view(other_dims, dim_size)
            
            # Calculate new strides
            new_strides = x_2d.stride()
            new_dim_stride = new_strides[1]  # dim is now the last dimension
            
            # Grid: (other_dims, min(dim_size, 1024))
            grid = lambda meta: (other_dims, min(dim_size, 1024))
            reverse_cumsum_fused_kernel[grid](
                x_2d, out_2d, other_dims, dim_size, new_dim_stride,
                BLOCK_SIZE=BLOCK_SIZE_DIM
            )
            
        return out


def triton_reverse_cumsum(x, dim):
    return TritonReverseCumsumFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension.
    Uses custom Triton kernels for improved performance.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)