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
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute offsets for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load predictions and targets
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference: (pred - target)^2
    diff = pred - target
    sq_diff = diff * diff
    
    # Mask the squared differences to ensure elements outside bounds don't contribute to the sum
    sq_diff = tl.where(mask, sq_diff, 0.0)
    
    # Sum the squared differences within the block
    block_sum = tl.sum(sq_diff, axis=0)
    
    # Store the partial sum in the output buffer
    tl.store(out_ptr + pid, block_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    # Ensure tensors are contiguous and on CUDA
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Number of blocks needed for the grid
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Buffer to store partial sums from each block
    partial_sums = torch.empty(grid_size, device=predictions.device, dtype=torch.float32)
    
    # Launch the Triton kernel
    mse_partial_sum_kernel[triton.Language.grid(grid_size)](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction of partial sums using PyTorch (highly optimized for small tensors)
    total_sum = torch.sum(partial_sums)
    
    # Compute the mean
    return total_sum / n_elements

class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with fused Triton implementation
        return triton_mse(predictions, targets)