import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr, target_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Identify the program ID
    pid = tl.program_id(0)
    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to handle boundary conditions for the last block
    mask = offsets < n_elements

    # Load the predictions and targets
    p = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference: (p - t)^2
    diff = p - t
    sq = diff * diff
    
    # Sum the squared differences within the block
    # Multiply by mask to ensure out-of-bounds elements do not contribute to the sum
    block_sum = tl.sum(sq * mask, axis=0)
    
    # Atomically add the block's local sum to the global accumulator
    tl.atomic_add(out_ptr, block_sum)

def triton_mse(predictions, targets):
    """
    Fuses subtraction, squaring, and global summation into a single Triton kernel
    to minimize memory bandwidth usage and avoid intermediate tensor allocations.
    """
    # Ensure the inputs are contiguous on GPU
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    # Initialize a zero scalar tensor to store the global sum of squared errors
    out = torch.zeros((), device=predictions.device, dtype=torch.float32)

    # Tunable block size
    BLOCK_SIZE = 1024
    # Calculate the number of blocks needed to cover all elements
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    mse_kernel[grid](
        predictions, targets, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Divide the global sum by the total number of elements to compute the mean
    return out / n_elements

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks,
    optimized with a fused Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace the PyTorch MSE calculation with the fused Triton implementation
        return triton_mse(predictions, targets)