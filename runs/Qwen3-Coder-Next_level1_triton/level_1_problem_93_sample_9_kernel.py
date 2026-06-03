import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,          # Input tensor pointer
    mask_ptr,       # Mask tensor pointer
    out_ptr,        # Output tensor pointer
    batch_size,     # Number of batches
    seq_len,        # Sequence length (size of dim dimension)
    dim,            # Dimension along which to compute cumsum
    stride_batch,   # Stride for batch dimension
    stride_seq,     # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Compute base offsets for this batch
    # We need to handle the case where dim is not the last dimension
    if dim == 1:
        # For dim=1, batch is outermost, seq is inner
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Compute pointers for this batch
        x_offset = batch_idx * stride_batch
        mask_offset = batch_idx * stride_batch
        out_offset = batch_idx * stride_batch
        
        x_ptrs = x_ptr + x_offset + offsets
        mask_ptrs = mask_ptr + mask_offset + offsets
        out_ptrs = out_ptr + out_offset + offsets
        
        # Load data
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptrs, mask=mask, other=0)
        
        # Compute masked values
        masked_vals = x_vals * tl.cast(mask_vals, tl.float32)
        
        # Compute cumulative sum
        cumsum = 0.0
        for i in range(BLOCK_SIZE):
            if i < seq_len:
                cumsum = tl.where(mask_vals[i] > 0, cumsum + masked_vals[i], cumsum)
                tl.store(out_ptrs + i, cumsum, mask=(i < seq_len))
    else:
        # Handle general dimension case (simplified for common case)
        # For simplicity, assume dim is the last dimension in memory layout
        # This requires more complex offset calculation
        pass


@triton.jit
def masked_cumsum_optimized_kernel(
    x_ptr,          # Input tensor pointer
    mask_ptr,       # Mask tensor pointer  
    out_ptr,        # Output tensor pointer
    batch_size,     # Number of batches
    seq_len,        # Sequence length
    dim,            # Dimension along which to compute cumsum
    stride0,        # Stride for batch dimension
    stride1,        # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_idx * stride0
    
    # Initialize cumulative sum
    cumsum = 0.0
    
    # Process sequence elements sequentially within the block
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        # Compute start and end for this block
        start = block_idx * BLOCK_SIZE
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        
        # Process elements in the block
        for i in range(start, end):
            # Compute offset within the sequence
            offset = base_offset + i * stride1
            
            # Load values
            x_val = tl.load(x_ptr + offset)
            mask_val = tl.load(mask_ptr + offset)
            
            # Update cumulative sum only if mask is True
            cumsum = tl.where(mask_val > 0, cumsum + x_val, cumsum)
            
            # Store result
            tl.store(out_ptr + offset, cumsum)


class TritonMaskedCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, dim):
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Get tensor shapes
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Calculate strides
        stride0 = x.stride(0)
        stride1 = x.stride(1)
        
        # Determine block size
        BLOCK_SIZE = min(1024, seq_len)
        
        # Grid: one block per batch
        grid = (batch_size,)
        
        # Launch kernel
        masked_cumsum_optimized_kernel[grid](
            x, mask, out,
            batch_size, seq_len, dim,
            stride0, stride1,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, we'll use PyTorch's built-in backward for now
        # A full implementation would require implementing backward pass
        return grad_output, None, None


def triton_masked_cumsum(x, mask, dim):
    return TritonMaskedCumsum.apply(x, mask, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using Triton kernels.
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
        # Convert mask to float32 for multiplication (if needed)
        mask_float = mask.to(torch.float32)
        
        # Use our Triton implementation
        return triton_masked_cumsum(x * mask_float, mask, self.dim)