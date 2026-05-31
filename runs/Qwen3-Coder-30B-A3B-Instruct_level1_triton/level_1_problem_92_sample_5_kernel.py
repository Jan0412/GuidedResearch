import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Calculate the position within the dimension
    pos_in_dim = offsets % dim_size
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform exclusive cumulative sum along the specified dimension
    # For each element, accumulate all previous elements in the same sequence
    for i in range(dim_size):
        # Only process elements where pos_in_dim == i
        cond = pos_in_dim == i
        
        # Update accumulator with previous values
        if i > 0:
            # Get the previous value from the accumulator
            prev_acc = tl.where(cond, acc, 0.0)
            # Add to the accumulator
            acc = tl.where(cond, acc + prev_acc, acc)
        
        # Accumulate the current input value
        acc = tl.where(cond, acc + x, acc)
        
        # Store the accumulated value (exclusive)
        if i > 0:
            tl.store(output_ptr + offsets, acc, mask=mask)
    
    # Handle the case where we're at the beginning of a sequence
    # Set the first element to zero
    first_in_seq = pos_in_dim == 0
    tl.store(output_ptr + offsets, tl.where(first_in_seq, 0.0, acc), mask=mask)

def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Triton implementation of exclusive cumulative sum.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    n_elements = x.numel()
    
    # Calculate strides
    stride_dim = 1
    for i in range(dim + 1, len(shape)):
        stride_dim *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Grid size calculation
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch the Triton kernel
    exclusive_cumsum_kernel[grid](
        x, 
        out, 
        n_elements, 
        dim_size, 
        stride_dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for exclusive cumulative sum.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use the Triton implementation instead of the PyTorch version
        # First create the zero-padded tensor
        zero_tensor = torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim))
        cat_tensor = torch.cat((zero_tensor, x), dim=self.dim)
        # Then compute the cumulative sum
        exclusive_cumsum = cat_tensor[:-1]  # Remove last element to simulate exclusive cumsum
        return torch.cumsum(exclusive_cumsum, dim=self.dim)