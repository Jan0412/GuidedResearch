import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_partial_sum_kernel(
    predictions_ptr,
    targets_ptr,
    partial_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Identify the block index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = p - t
    sq_diff = diff * diff
    
    # Mask out values beyond n_elements before summing
    sq_diff = tl.where(mask, sq_diff, 0.0)
    
    # Block-level reduction
    block_sum = tl.sum(sq_diff, axis=0)
    
    # Store the partial sum for this block
    tl.store(partial_sum_ptr + pid, block_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes Mean Squared Error using a fused Triton kernel for 
    element-wise operations and a partial reduction.
    """
    # Ensure tensors are on CUDA and contiguous
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Buffer to store partial sums from each block
    partial_sums = torch.empty(num_blocks, device=predictions.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (num_blocks,)
    mse_partial_sum_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction using PyTorch (efficient for the reduced size of partial_sums)
    total_sum = torch.sum(partial_sums)
    return total_sum / n_elements

class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with Triton implementation
        return triton_mse(predictions, targets)