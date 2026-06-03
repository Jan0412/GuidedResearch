import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    
    # Calculate starting offset for this batch
    # For dim=1, each batch is contiguous
    if dim == 1:
        offset = batch_idx * seq_len
    else:
        # For other dimensions, calculate accordingly
        offset = batch_idx * seq_len
    
    # Pointer to current batch
    x_batch_ptr = x_ptr + offset
    out_batch_ptr = out_ptr + offset
    
    # We'll process in chunks of BLOCK_SIZE
    # For exclusive cumsum, we need to track the running sum
    # and shift it by one position
    
    # Process the sequence in blocks
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First pass: compute cumulative sums for each block
    # We'll store intermediate sums in local memory
    for block_idx in range(num_blocks):
        start_idx = block_idx * BLOCK_SIZE
        block_size = min(BLOCK_SIZE, seq_len - start_idx)
        
        # Accumulate sum for this block
        cumsum = 0.0
        for i in range(block_size):
            idx = start_idx + i
            x_val = tl.load(x_batch_ptr + idx)
            cumsum += x_val
        
        # Store the cumulative sum at the end of the block
        if block_idx > 0:
            tl.store(out_batch_ptr + seq_len + block_idx - 1, cumsum)
    
    # Second pass: compute exclusive cumsum
    # For each position i, exclusive_cumsum[i] = sum(x[0:i])
    # We can compute this efficiently by maintaining a running sum
    
    running_sum = 0.0
    for i in range(seq_len):
        # Store the current running sum (exclusive cumsum)
        tl.store(out_batch_ptr + i, running_sum)
        # Update running sum with current element
        x_val = tl.load(x_batch_ptr + i)
        running_sum += x_val


@triton.jit
def exclusive_cumsum_kernel_fused(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for exclusive cumsum that processes each batch in parallel."""
    # Each program handles one batch
    batch_idx = tl.program_id(0)
    
    # Calculate the stride for the dimension
    # Assuming dim=1 for the given input shape
    stride = seq_len
    
    # Pointer to the start of this batch
    x_offset = batch_idx * stride
    out_offset = batch_idx * stride
    
    x_ptr_batch = x_ptr + x_offset
    out_ptr_batch = out_ptr + out_offset
    
    # Compute exclusive cumulative sum
    # exclusive_cumsum[i] = sum(x[0:i])
    cumsum = 0.0
    
    # Process elements sequentially within the batch
    for i in range(seq_len):
        # Store the current cumsum (which is sum of all previous elements)
        tl.store(out_ptr_batch + i, cumsum)
        # Add current element to cumsum
        x_val = tl.load(x_ptr_batch + i)
        cumsum += x_val


class TritonExclusiveCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Create output tensor
        out = torch.empty_like(x)
        
        batch_size = x.size(0)
        seq_len = x.size(1) if dim == 1 else x.size(0)
        
        # For the given architecture, dim is always 1 and input shape is (batch_size, seq_len)
        # So we'll optimize for dim=1
        if dim == 1:
            grid = (batch_size,)
            BLOCK_SIZE = 256
            
            exclusive_cumsum_kernel_fused[grid](
                x, out, batch_size, seq_len, dim,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # Fallback for other dimensions - use PyTorch implementation
            # This is not optimal but handles general cases
            exclusive_cumsum = torch.cat((torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x), dim=dim)[:-1]
            out = torch.cumsum(exclusive_cumsum, dim=dim)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For backward pass, we can use PyTorch's implementation
        # The gradient of exclusive cumsum is the cumulative sum of gradients
        # but we need to handle the exclusive nature
        grad_input = torch.cumsum(grad_output, dim=ctx.dim)
        
        # Remove the last element and prepend zero to match input shape
        grad_input = torch.cat((torch.zeros_like(grad_input.select(ctx.dim, 0).unsqueeze(ctx.dim)), 
                               grad_input), dim=ctx.dim)[:-1]
        
        return grad_input, None


def triton_exclusive_cumsum(x, dim):
    return TritonExclusiveCumsumFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using Triton kernel.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)