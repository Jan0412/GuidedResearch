import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    pred_ptr, target_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    p = tl.load(pred_ptr + offsets, mask=mask, other=1.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # To avoid NaNs during evaluation of the where clause (as some hardware evaluates both branches),
    # we use safe versions of the tensors for the log operation.
    t_safe = tl.where(t > 0, t, 1.0)
    p_safe = tl.where(p > 0, p, 1.0)
    
    # KL Divergence formula: target * (log(target) - log(prediction))
    # We handle target=0 explicitly because 0 * log(0) = 0.
    val = tl.where(t > 0, t * (tl.log(t_safe) - tl.log(p_safe)), 0.0)
    
    # Mask out elements outside the range before summing
    val = tl.where(mask, val, 0.0)
    
    # Sum the current block and store the partial result
    block_sum = tl.sum(val, axis=0)
    tl.store(out_ptr + pid, block_sum)

def triton_kl_div(predictions, targets):
    """
    Optimized KL Divergence implementation using Triton.
    Fuses log and element-wise multiplication into a single kernel to avoid
    intermediate memory allocations and reduce memory passes.
    """
    # Get dimensions
    N = predictions.shape[0]
    n_elements = predictions.numel()
    
    # Ensure tensors are contiguous on GPU
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks)
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate buffer for partial sums from each block
    partial_sums = torch.empty(grid_size, device=predictions.device, dtype=torch.float32)

    # Launch the Triton kernel
    kl_div_kernel[grid_size](
        predictions, targets, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Final reduction: sum all partial sums and divide by batch size (reduction='batchmean')
    return torch.sum(partial_sums) / N

class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions,
    optimized with a custom Triton kernel for fusion and speedup.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
        # with our fused Triton implementation.
        return triton_kl_div(predictions, targets)