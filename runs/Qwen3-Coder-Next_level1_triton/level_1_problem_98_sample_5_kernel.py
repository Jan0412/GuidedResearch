import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    log_preds_ptr,  # Pointer to log(predictions)
    targets_ptr,    # Pointer to targets
    out_ptr,        # Pointer to output (scalar result)
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulator for reduction
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load log(predictions) and targets
        log_preds = tl.load(log_preds_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute KL divergence: targets * (log(targets) - log(predictions))
        # Use safe computation: targets * log(targets) - targets * log(predictions)
        # Since targets is already a distribution (sums to 1), we compute:
        # KL = sum(targets * (log(targets) - log_preds))
        
        # Compute term1 = targets * log(targets) (entropy of targets)
        # But for numerical stability, we'll compute targets * (log(targets) - log_preds) directly
        term = targets * (tl.log(targets) - log_preds)
        
        # Accumulate sum
        sum_acc += term * mask
        
    # Reduction within block
    for i in range(BLOCK_SIZE // 2):
        sum_acc = sum_acc + tl.roll(sum_acc, 1 << i)
    
    # Get first element as block sum
    block_sum = sum_acc[0]
    
    # Atomic add to output (since multiple blocks may run)
    tl.atomic_add(out_ptr, block_sum)


@triton.jit
def kl_div_batchmean_kernel(
    log_preds_ptr,  # Pointer to log(predictions)
    targets_ptr,    # Pointer to targets
    out_ptr,        # Pointer to output (scalar result)
    batch_size,     # Number of samples in batch
    dim_size,       # Size of the dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    batch_id = tl.program_id(0)
    
    # Compute start and end indices for this batch sample
    start_idx = batch_id * dim_size
    end_idx = start_idx + dim_size
    
    # Accumulator for this sample
    sample_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, dim_size, BLOCK_SIZE):
        offsets = start_idx + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        
        # Load log(predictions) and targets
        log_preds = tl.load(log_preds_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute KL divergence for this sample: targets * (log(targets) - log(predictions))
        term = targets * (tl.log(targets) - log_preds)
        sample_sum += term * mask
    
    # Reduction within block
    for i in range(BLOCK_SIZE // 2):
        sample_sum = sample_sum + tl.roll(sample_sum, 1 << i)
    
    # Get first element as block sum
    sample_result = sample_sum[0]
    
    # Store result for this sample
    tl.store(out_ptr + batch_id, sample_result)


def triton_kl_div(log_preds: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence with batchmean reduction using Triton.
    
    Args:
        log_preds: log(predictions) tensor
        targets: targets tensor (must be same shape as log_preds)
    
    Returns:
        Scalar tensor with KL divergence (batchmean reduction)
    """
    assert log_preds.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert log_preds.shape == targets.shape, "Input tensors must have the same shape."
    
    log_preds = log_preds.contiguous()
    targets = targets.contiguous()
    
    # Get batch size and dimension size
    batch_size = log_preds.shape[0]
    dim_size = log_preds.shape[1] if len(log_preds.shape) > 1 else 1
    
    # For batchmean, we compute per-sample KL and then average
    # Create intermediate tensor to store per-sample KL values
    intermediate = torch.zeros(batch_size, device=log_preds.device, dtype=torch.float32)
    
    # Launch kernel with one block per batch sample
    BLOCK_SIZE = 128
    grid = (batch_size,)
    
    kl_div_batchmean_kernel[grid](
        log_preds, targets, intermediate, 
        batch_size, dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean of per-sample KL divergences
    result = intermediate.mean()
    
    return result


class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    Uses optimized Triton kernels for computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Compute log(predictions)
        log_preds = torch.log(predictions)
        # Use our Triton kernel for KL divergence with batchmean reduction
        return triton_kl_div(log_preds, targets)