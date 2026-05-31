import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate the starting position for this batch and sequence
    start_pos = batch_idx * seq_len + seq_idx * seq_len
    
    # Shared memory for reduction
    shared_sum = tl.shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)
    
    # Process elements in chunks
    total_sum = 0.0
    
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate actual offset
        offset = start_pos + i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offset < start_pos + seq_len
        
        # Load predictions and targets
        pred = tl.load(predictions_ptr + offset, mask=mask, other=1e-8)
        target = tl.load(targets_ptr + offset, mask=mask, other=0.0)
        
        # Compute kl divergence: target * log(target / pred)
        # We use the identity: target * log(target) - target * log(pred)
        # But since we're computing KL divergence, it's: target * log(target/pred)
        # Which equals: target * (log(target) - log(pred))
        # But we compute: target * log(target) - target * log(pred)
        # For numerical stability, we compute: target * log(target) - target * log(pred)
        # Handle edge cases where target or pred might be zero
        pred_safe = tl.where(pred > 1e-30, pred, 1e-30)
        target_safe = tl.where(target > 1e-30, target, 1e-30)
        
        # Compute kl divergence: target * log(target / pred)
        kl_val = target * tl.log(target_safe / pred_safe)
        
        # Accumulate sum
        total_sum += tl.sum(kl_val * mask)
    
    # Write to global memory
    if seq_idx == 0 and batch_idx < batch_size:
        tl.store(output_ptr + batch_idx, total_sum)

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Process one sequence at a time
    batch_idx = tl.program_id(0)
    seq_start = batch_idx * seq_len
    
    # Shared memory for max and sum computations
    shared_max = tl.shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)
    shared_sum = tl.shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)
    
    # First pass: find maximum value
    max_val = -float('inf')
    for i in range(0, seq_len, BLOCK_SIZE):
        offset = seq_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_start + seq_len
        val = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(val))
    
    # Broadcast max_val to all threads in the block
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))
    
    # Second pass: compute sum of exponentials
    sum_exp = 0.0
    for i in range(0, seq_len, BLOCK_SIZE):
        offset = seq_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_start + seq_len
        val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        exp_val = tl.exp(val - max_val)
        sum_exp += tl.sum(exp_val * mask)
    
    # Normalize and store results
    inv_sum = 1.0 / sum_exp
    for i in range(0, seq_len, BLOCK_SIZE):
        offset = seq_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_start + seq_len
        val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        log_val = val - max_val - tl.log(sum_exp)
        tl.store(output_ptr + offset, log_val, mask=mask)

def triton_kl_div(predictions, targets):
    """Custom Triton implementation of KL divergence"""
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    batch_size, seq_len = predictions.shape
    n_elements = batch_size * seq_len
    
    # Create output tensor
    out = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size, 1)
    
    # Launch kernel
    kl_div_kernel[grid](
        predictions,
        targets,
        out,
        n_elements,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean over batches
    return out.mean()

def triton_log_softmax(x):
    """Custom Triton implementation of log softmax"""
    assert x.is_cuda, "Tensor must be on CUDA"
    
    batch_size, seq_len = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x,
        out,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    Optimized with custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Apply log softmax to predictions using custom Triton kernel
        log_predictions = triton_log_softmax(predictions)
        
        # Compute KL divergence using custom Triton kernel
        # Note: In practice, we could fuse these operations further
        kl_div = triton_kl_div(log_predictions, targets)
        
        return kl_div