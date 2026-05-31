import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    p_ptr,  # Pointer to predictions
    t_ptr,  # Pointer to targets
    out_ptr,  # Pointer to partial sums
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID and grid information
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    
    # Grid-stride loop to handle very large tensors with a fixed number of blocks
    # This prevents the grid from becoming too large and reduces memory overhead for partial sums
    block_start = pid * BLOCK_SIZE
    stride = num_blocks * BLOCK_SIZE
    
    # Local accumulator for the sum of squared differences
    sum_val = 0.0
    
    # Iterate over the data in strides
    for i in range(0, (n_elements + stride - 1) // stride):
        offsets = block_start + i * stride + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        # Using other=0.0 is safe because (0.0 - 0.0)**2 = 0.0
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        t = tl.load(t_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared difference
        diff = p - t
        sq_diff = diff * diff
        
        # Sum the squared differences within the block
        sum_val += tl.sum(sq_diff, axis=0)
        
    # Store the partial sum for this block
    tl.store(out_ptr + pid, sum_val)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Optimized MSE loss using Triton.
    Fuses subtraction, squaring, and partial reduction into a single kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Flatten tensors to handle any input shape
    p = predictions.reshape(-1).contiguous()
    t = targets.reshape(-1).contiguous()
    n_elements = p.numel()
    
    # Hyperparameters
    BLOCK_SIZE = 1024
    NUM_BLOCKS = 1024 # Fixed number of blocks to manage partial sums array size
    
    # Buffer to store partial sums from each block
    partial_sums = torch.empty(NUM_BLOCKS, device=p.device, dtype=torch.float32)
    
    # Launch the kernel
    grid = (NUM_BLOCKS,)
    mse_kernel[grid](
        p, t, partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction: sum partial sums and divide by total elements
    # torch.sum is extremely efficient for a small number of elements (1024)
    total_sum = torch.sum(partial_sums)
    return total_sum / n_elements

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks,
    optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use the optimized Triton implementation instead of torch.mean((p - t)**2)
        return triton_mse(predictions, targets)