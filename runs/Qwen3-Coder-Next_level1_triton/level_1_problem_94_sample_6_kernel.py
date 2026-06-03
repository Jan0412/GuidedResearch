import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the global index of the thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute squared differences
    diff = predictions - targets
    squared_diff = diff * diff

    # Accumulate in a block for partial sums
    block_sum = tl.sum(squared_diff, axis=0)

    # Store partial sum to the output buffer
    tl.atomic_add(output_ptr, block_sum)


@triton.jit
def finalize_mse_kernel(
    partial_sums_ptr,  # Pointer to partial sums buffer
    output_ptr,        # Pointer to final output (scalar)
    n_partial_sums,    # Number of partial sums (number of blocks)
    total_elements,    # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the global index of the thread
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums

    # Load partial sums
    partial_sum = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)

    # Accumulate all partial sums
    total_sum = tl.sum(partial_sum, axis=0)
    
    # Finalize: divide by total elements to get mean
    if pid == 0:
        mean_value = total_sum / total_elements
        tl.store(output_ptr, mean_value)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes Mean Squared Error loss using Triton kernels.
    
    This implementation:
    1. Uses a first kernel to compute partial sums of squared differences in parallel
    2. Uses a second kernel to finalize the mean calculation
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate buffer for partial sums (one per block)
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=predictions.device)
    
    # Allocate output tensor (scalar)
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Launch first kernel to compute partial sums
    grid = (num_blocks,)
    mse_loss_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch second kernel to finalize the mean calculation
    grid_final = (1,)
    finalize_mse_kernel[grid_final](
        partial_sums, 
        output, 
        num_blocks, 
        n_elements,
        BLOCK_SIZE=128
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use the Triton-based MSE loss instead of PyTorch's native operation
        return triton_mse_loss(predictions, targets)