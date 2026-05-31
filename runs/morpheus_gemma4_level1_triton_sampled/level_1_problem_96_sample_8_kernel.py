import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute smooth L1 loss element-wise
    # Formula: 0.5 * (x-y)^2 if |x-y| < 1 else |x-y| - 0.5
    diff = x - y
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    # Mask the loss to ensure out-of-bounds elements don't contribute to the sum
    loss = tl.where(mask, loss, 0.0)
    
    # Local reduction within the block
    block_sum = tl.sum(loss, axis=0)
    
    # Store the partial sum for this block
    tl.store(out_ptr + pid, block_sum)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton implementation of smooth_l1_loss with reduction.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure inputs are contiguous and float32
    predictions = predictions.contiguous().float()
    targets = targets.contiguous().float()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    
    # Output buffer for partial sums (one per block)
    out_partial_sums = torch.empty(grid_size, device=predictions.device, dtype=torch.float32)
    
    # Launch kernel
    smooth_l1_kernel[grid](
        predictions, 
        targets, 
        out_partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction: sum of partial sums divided by total elements
    total_loss = torch.sum(out_partial_sums)
    return total_loss / n_elements

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks,
    optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)