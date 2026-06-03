import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    stride,  # Stride along the dimension we're cumsumming
    n_cols,  # Number of elements along the cumsum dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    row_start = row_idx * stride
    
    # We'll compute exclusive cumsum: out[i] = sum(x[0:i])
    # For i=0, out[0] = 0
    # For i>0, out[i] = x[0] + x[1] + ... + x[i-1]
    
    # First pass: compute inclusive cumsum and store in a temporary location
    # But since we need exclusive, we can shift the result
    
    # Initialize running sum
    cumsum = 0.0
    
    # Process in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load current element
        x_offset = row_start + offsets * stride
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Store the current cumsum (which is exclusive for position 'start')
        out_offset = row_start + offsets * stride
        tl.store(out_ptr + out_offset, cumsum, mask=mask)
        
        # Update cumsum with current block
        cumsum = cumsum + tl.sum(x, axis=0)


class TritonExclusiveCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        shape = x.shape
        n_rows = 1
        for i in range(len(shape)):
            if i != dim:
                n_rows *= shape[i]
        
        n_cols = shape[dim]
        stride = 1
        for i in range(dim + 1, len(shape)):
            stride *= shape[i]
        
        # Set block size
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
        
        # Grid: one block per row
        grid = (n_rows,)
        
        # Launch kernel
        exclusive_cumsum_kernel[grid](x, out, stride, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For exclusive cumsum: out[i] = sum(x[0:i])
        # grad_x[j] = sum of grad_output[i] for all i > j
        # This is equivalent to exclusive cumsum of reversed grad_output, then reversed again
        
        # Actually, let's think about this more carefully:
        # out[i] = x[0] + x[1] + ... + x[i-1]
        # So ∂out[i]/∂x[j] = 1 if j < i, else 0
        # Therefore grad_x[j] = sum_{i>j} grad_output[i]
        
        # This is the reverse exclusive cumsum of grad_output
        # Compute inclusive cumsum from right, then shift
        
        dim = ctx.dim
        grad_output = grad_output.contiguous()
        
        shape = grad_output.shape
        n_rows = 1
        for i in range(len(shape)):
            if i != dim:
                n_rows *= shape[i]
        
        n_cols = shape[dim]
        stride = 1
        for i in range(dim + 1, len(shape)):
            stride *= shape[i]
        
        # For backward pass, we need to compute exclusive cumsum from the right
        # grad_x[j] = grad_output[j+1] + grad_output[j+2] + ... + grad_output[n-1]
        
        # We can do this by reversing, doing forward exclusive cumsum, then reversing back
        
        # Actually, let's implement it directly with a kernel
        # For simplicity, we'll use PyTorch's cumsum for backward since it's well optimized
        # But to keep everything in Triton, let's implement a backward kernel
        
        # Create output for backward
        grad_input = torch.empty_like(grad_output)
        
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
        grid = (n_rows,)
        
        # Reverse cumsum kernel
        exclusive_cumsum_backward_kernel[grid](grad_output, grad_input, stride, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        return grad_input, None


@triton.jit
def exclusive_cumsum_backward_kernel(
    grad_ptr,  # Gradient pointer
    grad_input_ptr,  # Gradient input pointer (output)
    stride,  # Stride along the dimension we're cumsumming
    n_cols,  # Number of elements along the cumsum dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    row_start = row_idx * stride
    
    # We want: grad_input[j] = sum_{i>j} grad_output[i]
    # This is the exclusive cumsum from the right
    
    # First, compute total sum
    total_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        grad_offset = row_start + offsets * stride
        grad = tl.load(grad_ptr + grad_offset, mask=mask, other=0.0)
        total_sum = total_sum + tl.sum(grad, axis=0)
    
    # Now compute exclusive cumsum from the left, and subtract from total
    cumsum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        grad_offset = row_start + offsets * stride
        grad = tl.load(grad_ptr + grad_offset, mask=mask, other=0.0)
        
        # grad_input = total_sum - cumsum - grad (exclusive, so exclude current)
        # But actually: grad_input[j] = total_sum - cumsum_at_j (where cumsum_at_j includes j)
        # So we need to store cumsum first, then update
        
        tl.store(grad_input_ptr + grad_offset, total_sum - cumsum, mask=mask)
        
        # Update cumsum with current block
        cumsum = cumsum + tl.sum(grad, axis=0)


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
        # Use the custom Triton function for exclusive cumsum
        return TritonExclusiveCumsumFunction.apply(x, self.dim)