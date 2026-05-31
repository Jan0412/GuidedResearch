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
    shared_mem = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Process elements in chunks
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate actual offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_len
        
        # Load predictions and targets
        pred = tl.load(predictions_ptr + start_pos + offset, mask=mask, other=0.0)
        target = tl.load(targets_ptr + start_pos + offset, mask=mask, other=0.0)
        
        # Compute KL divergence: sum(target * (log(target) - log(predictions)))
        # We'll compute log(target) and log(predictions) separately
        pred_log = tl.where(pred > 0, tl.log(pred), 0.0)
        target_log = tl.where(target > 0, tl.log(target), 0.0)
        
        # Compute kl_div contribution
        kl_contribution = target * (target_log - pred_log)
        
        # Store in shared memory for reduction
        tl.store(shared_mem + offset, kl_contribution, mask=mask)
        
        # Synchronize threads before reduction
        tl.sync()
        
        # Reduction within block
        if i == 0:
            # Initialize with first element
            acc = shared_mem[0] if 0 < seq_len else 0.0
            for j in range(1, min(BLOCK_SIZE, seq_len)):
                acc += shared_mem[j]
        else:
            # For subsequent chunks, just add the values
            acc = 0.0
            for j in range(min(BLOCK_SIZE, seq_len - i)):
                acc += shared_mem[j]
        
        # Reduce across all blocks in this batch/sequence
        if i == 0:
            # Store final result
            tl.store(output_ptr + batch_idx, acc, mask=batch_idx < batch_size)

# Optimized version using proper reduction
@triton.jit
def kl_div_kernel_optimized(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_idx * seq_len
    
    # Shared memory for reduction
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process elements in chunks
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate actual offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_len
        
        # Load predictions and targets
        pred = tl.load(predictions_ptr + base_offset + offset, mask=mask, other=0.0)
        target = tl.load(targets_ptr + base_offset + offset, mask=mask, other=0.0)
        
        # Compute KL divergence: target * (log(target) - log(predictions))
        pred_log = tl.where(pred > 0, tl.log(pred), 0.0)
        target_log = tl.where(target > 0, tl.log(target), 0.0)
        
        # Compute kl_div contribution
        kl_contribution = target * (target_log - pred_log)
        
        # Reduce within this chunk
        local_sum = tl.sum(kl_contribution, axis=0)
        acc += local_sum
    
    # Store final result
    tl.store(output_ptr + batch_idx, acc, mask=batch_idx < batch_size)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton implementation of KL divergence computation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, seq_len = predictions.shape
    
    # Prepare output tensor
    out = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size, 1)
    
    # Launch kernel
    kl_div_kernel_optimized[grid](
        predictions,
        targets,
        out,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean over batch dimension
    return out.mean()

class ModelNew(nn.Module):
    """
    An optimized model for computing Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our Triton kernel instead of PyTorch's kl_div
        kl_div_result = triton_kl_div(predictions, targets)
        return kl_div_result