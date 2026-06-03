import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_divergence_kernel(
    predictions_ptr,  # Pointer to predictions (P)
    targets_ptr,      # Pointer to targets (Q)
    output_ptr,       # Pointer to output
    batch_size,       # Batch size
    dim,              # Dimension along which to compute KL divergence
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    
    # Compute starting offset for this batch
    offset = batch_idx * dim
    
    # Create block for dimension
    dim_offsets = tl.arange(0, DIM_BLOCK)
    mask = dim_offsets < dim
    
    # Load predictions and targets for this batch
    preds = tl.load(predictions_ptr + offset + dim_offsets, mask=mask, other=0.0)
    tgts = tl.load(targets_ptr + offset + dim_offsets, mask=mask, other=0.0)
    
    # Compute log predictions for numerical stability
    # Use log_softmax style computation for stability
    log_preds = tl.log(preds)
    
    # Compute KL divergence: P * (log(P) - log(Q)) = P * log(P/Q)
    # Which is: P * (log(P) - log(Q))
    kl_contrib = preds * (log_preds - tl.log(tgts))
    
    # Sum over dimension
    kl_sum = tl.sum(kl_contrib, axis=0)
    
    # Store result
    tl.store(output_ptr + batch_idx, kl_sum)


@triton.jit
def reduction_kernel(
    input_ptr,        # Pointer to batch-wise KL sums
    output_ptr,       # Pointer to final mean output
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Load all batch sums
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    batch_sums = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sum all batch values
    total_sum = tl.sum(batch_sums, axis=0)
    
    # Compute mean (batchmean reduction)
    mean_val = total_sum / batch_size
    
    # Store result
    tl.store(output_ptr, mean_val)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence with batchmean reduction using Triton kernels.
    KL(P||Q) = sum(P * log(P/Q)) / batch_size
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Input shapes must match."
    assert predictions.dim() == 2, "Expected 2D tensors."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, dim = predictions.shape
    
    # Allocate output for batch sums
    batch_sums = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # Configure kernel dimensions
    BLOCK_SIZE = 128
    # Use a reasonable block size for the dimension dimension
    DIM_BLOCK = min(256, triton.next_power_of_2(dim))
    
    # Launch kernel for each batch item
    grid = (batch_size,)
    
    # Compute KL divergence for each batch item
    kl_divergence_kernel[grid](
        predictions, targets, batch_sums,
        batch_size, dim, batch_size * dim,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM_BLOCK=DIM_BLOCK,
    )
    
    # Compute mean across batch
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    reduction_grid = (1,)
    
    # Configure reduction kernel
    reduction_kernel[reduction_grid](
        batch_sums, output,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Ensure inputs are in the right format
        # Note: The user should provide already normalized predictions and targets
        # (summing to 1 along the specified dimension)
        return triton_kl_div(predictions, targets)