import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_partial_sum_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute partial sums of squared differences.
    Each block processes a chunk of the input tensors and stores its local sum.
    """
    # Get the program ID to determine the data chunk this block handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference: (pred - target)^2
    diff = pred - target
    sq = diff * diff
    
    # Mask out-of-bounds elements to 0 before performing the block-level sum
    sq = tl.where(mask, sq, 0.0)
    local_sum = tl.sum(sq, axis=0)
    
    # Store the partial sum for this block into the output buffer
    tl.store(out_ptr + pid, local_sum)

def triton_mse(predictions, targets):
    """
    Wrapper function that manages memory, grid calculation, and final reduction.
    """
    # Ensure inputs are contiguous on the GPU for efficient pointer arithmetic
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    # BLOCK_SIZE is tuned for FP32 throughput; 4096 is generally efficient on modern GPUs
    BLOCK_SIZE = 4096
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate a tensor to hold partial sums from each Triton block
    # This avoids atomic contention on a single scalar and improves performance
    partial_sums = torch.empty((grid_size,), device=predictions.device, dtype=torch.float32)
    
    # Define the grid (one block per chunk of BLOCK_SIZE elements)
    grid = (grid_size,)
    
    # Launch the fused Triton kernel
    mse_partial_sum_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction: sum all partial sums and divide by total elements to get the mean
    # torch.sum is highly optimized for this final small-scale reduction
    return torch.sum(partial_sums) / n_elements

class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using a fused Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with the fused Triton implementation
        return triton_mse(predictions, targets)