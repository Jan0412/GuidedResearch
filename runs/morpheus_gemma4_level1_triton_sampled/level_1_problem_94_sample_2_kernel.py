import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr, 
    target_ptr, 
    block_sums_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    p = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute squared difference: (p - t)^2
    diff = p - t
    sq_diff = diff * diff
    
    # Mask out elements that are out of bounds to avoid affecting the sum
    sq_diff = tl.where(mask, sq_diff, 0.0)

    # Sum the squared differences within the block
    local_sum = tl.sum(sq_diff, axis=0)

    # Store the partial sum in the block_sums tensor
    tl.store(block_sums_ptr + pid, local_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton-optimized Mean Squared Error computation.
    Fuses subtraction, squaring, and partial summation into a single kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Flatten and ensure tensors are contiguous for the kernel
    p_flat = predictions.view(-1).contiguous()
    t_flat = targets.view(-1).contiguous()
    
    n_elements = p_flat.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Buffer to store partial sums from each block
    block_sums = torch.empty(grid_size, dtype=torch.float32, device=predictions.device)

    # Launch the Triton kernel
    grid = (grid_size,)
    mse_kernel[grid](
        p_flat, 
        t_flat, 
        block_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Final reduction: sum partial results and divide by total elements
    # PyTorch's sum is highly optimized for 1D tensors
    total_sum = torch.sum(block_sums)
    return total_sum / n_elements

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks,
    optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)