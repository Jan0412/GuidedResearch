import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    predictions_ptr,  # Pointer to predictions (P)
    targets_ptr,      # Pointer to targets (Q)
    output_ptr,       # Pointer to output
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: P * (log(P) - log(Q))
    # Use log_softmax style computation for numerical stability
    # KL = P * log(P/Q) = P * (log(P) - log(Q))
    log_predictions = tl.log(predictions)
    log_targets = tl.log(targets)
    
    # Compute KL divergence per element
    kl_elements = predictions * (log_predictions - log_targets)
    
    # Store result
    tl.store(output_ptr + offsets, kl_elements, mask=mask)


@triton.jit
def reduce_sum_kernel(
    input_ptr,       # Pointer to input
    output_ptr,      # Pointer to output (scalar)
    n_elements,      # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each block computes partial sum
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and sum
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + pid, sum_val)


@triton.jit
def final_reduction_kernel(
    partial_sums_ptr,  # Pointer to partial sums
    output_ptr,        # Pointer to final output (scalar)
    n_partial_sums,    # Number of partial sums
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    # Load partial sums
    partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums, axis=0)
    
    # Store final sum
    tl.store(output_ptr, total_sum)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence using Triton kernels.
    KL(P||Q) = sum(P * (log(P) - log(Q))) / batch_size
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Input shapes must match."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # First kernel: compute element-wise KL divergence
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Allocate intermediate storage for KL elements
    kl_elements = torch.empty_like(predictions)
    
    # Launch first kernel
    kl_div_kernel[grid](predictions, targets, kl_elements, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute sum using reduction
    # Calculate number of blocks needed for reduction
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_blocks > 1:
        # Allocate partial sums array
        partial_sums = torch.empty(num_blocks, dtype=predictions.dtype, device=predictions.device)
        
        # Launch reduction kernel
        reduce_sum_kernel[grid](kl_elements, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        # Final reduction
        final_block_size = min(256, num_blocks)
        final_grid = ((num_blocks + final_block_size - 1) // final_block_size,)
        
        # Allocate final output
        total_sum = torch.empty(1, dtype=predictions.dtype, device=predictions.device)
        final_reduction_kernel[final_grid](partial_sums, total_sum, num_blocks, BLOCK_SIZE=final_block_size)
        
        # Compute mean (batchmean reduction)
        result = total_sum / predictions.shape[0]
    else:
        # If only one block, just sum the elements directly
        total_sum = torch.sum(kl_elements)
        result = total_sum / predictions.shape[0]
    
    return result


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom Triton implementation of KL divergence
        return triton_kl_div(predictions, targets)