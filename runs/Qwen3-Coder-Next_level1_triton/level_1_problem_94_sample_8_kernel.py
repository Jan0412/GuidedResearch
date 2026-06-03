import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared error
    diff = predictions - targets
    squared_error = diff * diff
    
    # Store intermediate squared errors for reduction
    tl.store(output_ptr + offsets, squared_error, mask=mask)


@triton.jit
def sum_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each program handles one output element (sum of one block)
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and accumulate
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    # Store result
    tl.store(output_ptr + block_id, sum_val)


@triton.jit
def final_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Single block reduction for final sum
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    tl.store(output_ptr, sum_val)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Mean Squared Error loss using Triton kernels.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # First kernel: compute squared errors
    squared_errors = torch.empty_like(predictions)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mse_loss_kernel[grid](predictions, targets, squared_errors, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: reduce to single value
    # Calculate number of blocks needed for first reduction
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_blocks == 1:
        # Direct reduction if single block
        final_sum = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
        final_reduce_kernel[1](squared_errors, final_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Two-stage reduction
        reduced = torch.empty(num_blocks, device=predictions.device, dtype=predictions.dtype)
        sum_reduce_kernel[num_blocks](squared_errors, reduced, n_elements, BLOCK_SIZE=BLOCK_SIZE, NUM_BLOCKS=num_blocks)
        
        # Final reduction
        final_sum = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
        final_reduce_kernel[1](reduced, final_sum, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean
    mean_value = final_sum / n_elements
    return mean_value


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss for regression tasks
    using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)