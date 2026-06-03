import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_divergence_kernel(
    predictions_ptr,  # Pointer to predictions (P)
    targets_ptr,      # Pointer to targets (Q)
    output_ptr,       # Pointer to output
    batch_size,       # Number of batches
    n_elements,       # Total number of elements per batch
    BLOCK_SIZE: tl.constexpr
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    # Compute element offsets within the batch
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute pointer offsets for this batch
    batch_offset = batch_idx * n_elements
    
    # Load predictions and targets
    p = tl.load(predictions_ptr + batch_offset + offsets, mask=mask, other=0.0)
    q = tl.load(targets_ptr + batch_offset + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: p * (log(p) - log(q))
    # Add small epsilon for numerical stability
    epsilon = 1e-12
    p_safe = tl.maximum(p, epsilon)
    q_safe = tl.maximum(q, epsilon)
    
    log_p = tl.log(p_safe)
    log_q = tl.log(q_safe)
    
    kl = p * (log_p - log_q)
    
    # Store intermediate result for reduction
    tl.store(output_ptr + batch_offset + offsets, kl, mask=mask)


def triton_kl_divergence(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence with batchmean reduction using Triton.
    
    KL(P||Q) = mean(batch) of sum(elements) of P * (log(P) - log(Q))
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    n_elements = predictions.numel() // batch_size if batch_size > 0 else 0
    
    # Create intermediate buffer for batch-wise sums
    # For simplicity, we'll do the reduction on CPU/GPU for small batches
    # or use a two-step approach for larger ones
    output = torch.empty_like(predictions)
    
    BLOCK_SIZE = 256
    grid_x = batch_size
    grid_y = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel to compute element-wise KL divergence
    kl_divergence_kernel[grid_x, grid_y](
        predictions, targets, output,
        batch_size, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean across all elements (batchmean reduction)
    # Sum across the last dimension, then take mean across batch
    if batch_size > 0 and n_elements > 0:
        result = output.sum() / (batch_size * n_elements)
    else:
        result = torch.tensor(0.0, device=predictions.device)
    
    return result


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our optimized Triton kernel for KL divergence
        return triton_kl_divergence(predictions, targets)