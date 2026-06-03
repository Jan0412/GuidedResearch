import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    stride,  # Stride along the dimension of interest
    dim_size,  # Size of the dimension along which we compute cumsum
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" along the dimension we're not cumsumming
    # The program_id corresponds to the index in other dimensions
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    base_offset = row_idx * stride
    
    # We'll compute the exclusive cumsum in two passes for simplicity and correctness
    # First pass: compute the cumulative sum
    # Second pass: shift the results to get exclusive sum
    
    # For efficiency, we'll use a single kernel with a running sum approach
    # that stores the cumulative sum so far, then shifts it
    
    # Initialize running sum to 0
    running_sum = 0.0
    
    # Process each element in the dimension
    for i in range(dim_size):
        # Calculate offset for current position
        offset = base_offset + i * stride
        # Load current element
        val = tl.load(x_ptr + offset)
        # Store the current running sum (exclusive) to output
        tl.store(out_ptr + offset, running_sum)
        # Update running sum with current value
        running_sum = running_sum + val


class TritonExclusiveCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get tensor shape and calculate dimensions
        shape = x.shape
        dim_size = shape[dim]
        
        # Calculate stride along the cumsum dimension
        stride = 1
        for i in range(dim + 1, len(shape)):
            stride *= shape[i]
        
        # Calculate total number of "rows" to process
        # This is the number of elements in all dimensions except dim
        n_rows = x.numel() // dim_size
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Launch kernel with one block per row (one program per row)
        grid = (n_rows,)
        
        # Launch the Triton kernel
        exclusive_cumsum_kernel[grid](
            x, out, x.numel(), stride, dim_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For exclusive cumsum, the gradient is the cumulative sum of the output gradient
        # Since exclusive_cumsum(x)[i] = sum(x[:i]), then d/dx[j] exclusive_cumsum(x)[i] = 1 if j < i else 0
        # So the gradient w.r.t. x is the cumulative sum of grad_output from the end
        return torch.cumsum(grad_output.flip(dims=[ctx.dim]), dim=ctx.dim).flip(dims=[ctx.dim]), None


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