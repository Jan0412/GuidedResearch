import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    stride,  # Stride along the dimension we're computing cumsum
    BLOCK_SIZE: tl.constexpr,
    EVEN_ELEMENTS: tl.constexpr,
):
    # Get the program id for the batch dimension
    batch_id = tl.program_id(0)
    
    # Calculate offset for this batch
    offset = batch_id * stride
    
    # Initialize running sum to 0 for exclusive cumsum
    cumsum = 0.0
    
    # Process elements sequentially for each batch
    for i in range(0, stride, BLOCK_SIZE):
        # Calculate current position
        pos = offset + i
        
        # Load current element
        if EVEN_ELEMENTS:
            x = tl.load(x_ptr + pos)
        else:
            mask = (tl.arange(0, BLOCK_SIZE) < stride - i) if i + BLOCK_SIZE > stride else None
            if mask is not None:
                x = tl.load(x_ptr + pos, mask=mask)
            else:
                x = tl.load(x_ptr + pos)
        
        # Store the current cumsum (exclusive - doesn't include current element)
        if EVEN_ELEMENTS:
            tl.store(out_ptr + pos, cumsum)
        else:
            if mask is not None:
                tl.store(out_ptr + pos, cumsum, mask=mask)
            else:
                tl.store(out_ptr + pos, cumsum)
        
        # Update cumsum with current element
        cumsum = cumsum + x


@triton.jit
def exclusive_cumsum_kernel_optimized(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_batch,  # Number of batches
    seq_len,  # Sequence length (size along dim)
    stride,  # Stride along the dimension we're computing cumsum
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * stride
    
    # Initialize running sum to 0 for exclusive cumsum
    cumsum = 0.0
    
    # Process elements sequentially
    for i in range(0, seq_len):
        # Calculate current position
        pos = base_offset + i
        
        # Load current element
        x = tl.load(x_ptr + pos)
        
        # Store the current cumsum (exclusive - doesn't include current element)
        tl.store(out_ptr + pos, cumsum)
        
        # Update cumsum with current element
        cumsum = cumsum + x


class TritonExclusiveCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get shape information
        shape = x.shape
        dim_size = shape[dim]
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Calculate strides
        stride = 1
        for i in range(dim + 1, len(shape)):
            stride *= shape[i]
        
        # Calculate total number of batches (all dimensions except dim)
        n_batch = x.numel() // (dim_size * stride)
        
        # Set block size
        BLOCK_SIZE = 128
        
        # Calculate grid size
        grid = (n_batch,)
        
        # Launch kernel
        exclusive_cumsum_kernel_optimized[grid](
            x, out, n_batch, dim_size, stride,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For backward pass, we need to compute the gradient
        # Since forward is exclusive cumsum, backward is exclusive cumsum of gradients
        # But actually, the gradient of exclusive cumsum is just the exclusive cumsum of the output gradient
        # However, we need to be careful about the implementation
        return grad_output, None


def triton_exclusive_cumsum(x, dim):
    return TritonExclusiveCumsum.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using Triton kernels.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)