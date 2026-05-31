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
    
    # Each batch processes its own slice
    predictions_batch = predictions_ptr + batch_idx * seq_len
    targets_batch = targets_ptr + batch_idx * seq_len
    output_batch = output_ptr + batch_idx
    
    # Initialize accumulator for this batch
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Process elements in blocks
    block_start = 0
    while block_start < seq_len:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load predictions and targets
        pred = tl.load(predictions_batch + offsets, mask=mask, other=0.0)
        target = tl.load(targets_batch + offsets, mask=mask, other=0.0)
        
        # Compute KL divergence contribution for this element
        # kl_div = target * log(target / pred) = target * (log(target) - log(pred))
        # Handle numerical stability with small epsilon
        epsilon = 1e-8
        pred_safe = tl.maximum(pred, epsilon)
        target_safe = tl.maximum(target, epsilon)
        
        log_ratio = tl.log(target_safe) - tl.log(pred_safe)
        kl_contrib = target * log_ratio
        
        # Accumulate contribution
        acc += tl.sum(kl_contrib)
        
        block_start += BLOCK_SIZE
    
    # Store result for this batch
    tl.store(output_batch, acc)

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Process each batch independently
    batch_idx = tl.program_id(0)
    input_batch = input_ptr + batch_idx * seq_len
    output_batch = output_ptr + batch_idx * seq_len
    
    # First pass: find max value in the batch
    block_start = 0
    max_val = tl.zeros([1], dtype=tl.float32) - float('inf')
    
    while block_start < seq_len:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        input_vals = tl.load(input_batch + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(input_vals))
        block_start += BLOCK_SIZE
    
    # Second pass: compute log-sum-exp
    block_start = 0
    sum_exp = tl.zeros([1], dtype=tl.float32)
    
    while block_start < seq_len:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        input_vals = tl.load(input_batch + offsets, mask=mask, other=0.0)
        exp_val = tl.exp(input_vals - max_val)
        sum_exp += tl.sum(exp_val)
        block_start += BLOCK_SIZE
    
    # Third pass: compute final log-softmax values
    block_start = 0
    while block_start < seq_len:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        input_vals = tl.load(input_batch + offsets, mask=mask, other=0.0)
        output_vals = input_vals - max_val - tl.log(sum_exp)
        tl.store(output_batch + offsets, output_vals, mask=mask)
        block_start += BLOCK_SIZE

def triton_kl_div(predictions, targets):
    """
    Custom Triton implementation of KL divergence computation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    
    batch_size, seq_len = predictions.shape
    
    # Allocate output tensor
    output = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = lambda meta: (batch_size,)
    kl_div_kernel[grid](
        predictions,
        targets,
        output,
        predictions.numel(),
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean over batches
    return output.mean()

def triton_log_softmax(input_tensor):
    """
    Custom Triton implementation of log_softmax.
    """
    assert input_tensor.is_cuda, "Tensor must be on CUDA."
    
    batch_size, seq_len = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = lambda meta: (batch_size,)
    log_softmax_kernel[grid](
        input_tensor,
        output,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for KL divergence computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Apply log to predictions using Triton kernel
        log_predictions = triton_log_softmax(predictions)
        
        # Compute KL divergence using Triton kernel
        kl_loss = triton_kl_div(log_predictions, targets)
        
        return kl_loss