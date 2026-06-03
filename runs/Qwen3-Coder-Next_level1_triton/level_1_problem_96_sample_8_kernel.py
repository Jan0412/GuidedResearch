import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    delta: tl.constexpr = 1.0,
    reduction: tl.constexpr = 'mean',
    BLOCK_SIZE: tl.constexpr = 256,
):
    # Calculate global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    tgt = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute difference
    diff = pred - tgt
    abs_diff = tl.abs(diff)
    
    # Smooth L1 loss: 
    # if |x| < delta: 0.5 * x^2 / delta
    # else: |x| - 0.5 * delta
    cond = abs_diff < delta
    loss = tl.where(
        cond,
        0.5 * diff * diff / delta,
        abs_diff - 0.5 * delta
    )
    
    # Store intermediate loss values for reduction
    tl.store(output_ptr + offsets, loss, mask=mask)


@triton.jit
def sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    # This kernel performs parallel reduction to sum all elements
    # We'll do multiple passes if needed, but for simplicity we assume 
    # n_elements <= BLOCK_SIZE * num_programs
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Accumulate
    sum_val = tl.sum(x)
    
    # Store partial sum
    tl.store(output_ptr + pid, sum_val)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean'):
    """
    Triton implementation of Smooth L1 (Huber) Loss.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Create output tensor for loss values (same shape as input)
    loss_output = torch.empty_like(predictions)
    
    # Configure kernel launch
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch smooth L1 kernel
    smooth_l1_loss_kernel[grid](
        predictions, targets, loss_output, n_elements,
        delta=1.0, reduction=reduction, BLOCK_SIZE=BLOCK_SIZE
    )
    
    if reduction == 'mean':
        # For mean reduction, we need to compute the sum and divide
        # Use a two-level reduction approach
        
        # First, compute partial sums with a block size of 1024
        sum_block_size = 1024
        num_blocks = (n_elements + sum_block_size - 1) // sum_block_size
        
        # If n_elements is very large, we might need multiple passes
        # For simplicity, assume it fits in one pass for now
        partial_sums = torch.empty(num_blocks, device=predictions.device, dtype=predictions.dtype)
        
        sum_grid = lambda meta: (num_blocks,)
        sum_kernel[sum_grid](
            loss_output, partial_sums, n_elements,
            BLOCK_SIZE=sum_block_size
        )
        
        # Compute final sum and mean
        total_sum = torch.sum(partial_sums)
        return total_sum / n_elements
    elif reduction == 'sum':
        # For sum reduction, we can just sum the loss_output tensor
        return torch.sum(loss_output)
    else:  # reduction == 'none'
        return loss_output


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets, reduction='mean')