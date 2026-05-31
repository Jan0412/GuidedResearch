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
    beta,
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

    # Compute absolute difference
    diff = tl.abs(x - y)

    # Smooth L1 (Huber) Loss formula:
    # loss = 0.5 * diff^2 / beta if diff < beta else diff - 0.5 * beta
    loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    # Mask out elements beyond n_elements to avoid contributing to the sum
    loss = tl.where(mask, loss, 0.0)

    # Partial reduction within the block
    block_sum = tl.sum(loss, axis=0)
    
    # Store the partial sum for this block
    tl.store(out_ptr + pid, block_sum)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Triton implementation of smooth_l1_loss with mean reduction.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Grid size: number of blocks needed to cover all elements
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Buffer to store partial sums from each block
    partial_sums = torch.empty(grid_size, device=predictions.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (grid_size,)
    smooth_l1_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        beta, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction: sum partial results and divide by total elements for mean
    total_loss = torch.sum(partial_sums)
    return total_loss / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use the custom Triton implementation of smooth_l1_loss
        # PyTorch's default beta for smooth_l1_loss is 1.0
        return triton_smooth_l1_loss(predictions, targets, beta=1.0)