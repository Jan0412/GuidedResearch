import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each program handles a portion of the computation
    block_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Initialize accumulator for this block
    sum = tl.zeros([1], dtype=tl.float32)
    
    # Compute start and end indices for this block
    start_idx = block_id * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, n_elements)
    
    # Process elements in this block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (end_idx - start_idx)
    
    # Load predictions and targets
    pred = tl.load(predictions_ptr + start_idx + offsets, mask=mask, other=0.0)
    tgt = tl.load(targets_ptr + start_idx + offsets, mask=mask, other=0.0)
    
    # Compute squared difference and accumulate
    diff = pred - tgt
    sum += tl.sum(diff * diff)
    
    # Store partial sum to output array
    tl.store(output_ptr + block_id, sum)


@triton.jit
def final_reduce_kernel(
    partial_sums_ptr,  # Pointer to partial sums
    final_result_ptr,  # Pointer to final result
    n_partial_sums,    # Number of partial sums
    total_elements,    # Total number of elements for division
    BLOCK_SIZE: tl.constexpr,
):
    # Single block reduction to sum all partial sums
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    # Load partial sums
    partial = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partial)
    
    # Compute mean and store
    mean = total / tl.cast(total_elements, tl.float32)
    tl.store(final_result_ptr, mean)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Mean Squared Error loss using Triton kernels.
    
    Parameters:
        predictions: Predicted values tensor
        targets: Ground truth values tensor
    
    Returns:
        Scalar MSE loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # For large tensors, use block-based computation
    BLOCK_SIZE = 256
    # Calculate number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure reasonable number of blocks for efficient parallelism
    if num_blocks > 1024:
        num_blocks = 1024
    
    # Recalculate block size based on target number of blocks
    actual_block_size = (n_elements + num_blocks - 1) // num_blocks
    
    # Allocate partial sums buffer
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=predictions.device)
    
    # Launch first kernel
    grid = (num_blocks,)
    mse_loss_kernel[grid](
        predictions, targets, partial_sums,
        n_elements,
        BLOCK_SIZE=actual_block_size,
        NUM_BLOCKS=num_blocks
    )
    
    # Allocate final result
    result = torch.empty(1, dtype=torch.float32, device=predictions.device)
    
    # Launch reduction kernel
    final_reduce_kernel[(1,)](
        partial_sums, result,
        num_blocks, n_elements,
        BLOCK_SIZE=1024
    )
    
    return result


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom Triton kernel for MSE computation
        return triton_mse_loss(predictions, targets)