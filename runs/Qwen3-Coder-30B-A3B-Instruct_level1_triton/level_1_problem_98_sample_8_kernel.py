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
    sum_kl = tl.zeros((1,), dtype=tl.float32)
    
    # Process elements in chunks
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate offsets and mask
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load prediction and target values
        pred_vals = tl.load(predictions_batch + offsets, mask=mask, other=0.0)
        target_vals = tl.load(targets_batch + offsets, mask=mask, other=0.0)
        
        # Compute KL divergence contribution for each element
        # KL = sum(target * log(target/prediction))
        # We compute: target * log(target) - target * log(prediction)
        # But we avoid log(0) by checking for zeros
        
        # Compute target * log(target) - note: when target=0, this term is 0
        target_log_target = tl.where(target_vals > 0, target_vals * tl.log(target_vals), 0.0)
        
        # Compute target * log(prediction) - handle case where prediction is 0
        pred_log_pred = tl.where(pred_vals > 0, pred_vals * tl.log(pred_vals), 0.0)
        
        # Compute difference (KL contribution)
        kl_contrib = target_log_target - pred_log_pred
        
        # Accumulate
        sum_kl += tl.sum(kl_contrib)
    
    # Store the result for this batch
    tl.store(output_batch, sum_kl)

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Process one batch at a time
    batch_idx = tl.program_id(0)
    
    # Pointers for current batch
    input_batch = input_ptr + batch_idx * seq_len
    output_batch = output_ptr + batch_idx * seq_len
    
    # First pass: find max value
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    
    for i in range(0, seq_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        vals = tl.load(input_batch + offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Second pass: compute log-sum-exp and subtract max
    sum_exp = tl.zeros([1], dtype=tl.float32)
    
    for i in range(0, seq_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        vals = tl.load(input_batch + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals)
    
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Third pass: compute final log_softmax values
    for i in range(0, seq_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        vals = tl.load(input_batch + offsets, mask=mask, other=0.0)
        result = vals - log_sum_exp
        tl.store(output_batch + offsets, result, mask=mask)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton implementation of KL divergence computation
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    batch_size, seq_len = predictions.shape
    
    # Precompute log of predictions using a custom kernel
    log_predictions = torch.empty_like(predictions)
    
    # Use log_softmax kernel on predictions to compute log probabilities
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    # For the log_softmax kernel
    log_softmax_kernel[grid](
        predictions,
        log_predictions,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute KL divergence using another kernel
    # Output per batch
    output = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Launch KL divergence kernel
    kl_div_kernel[grid](
        log_predictions,
        targets,
        output,
        batch_size * seq_len,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean over all batches
    return torch.mean(output)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for KL divergence computation
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)

# Note: The original implementation computes torch.log(predictions) then applies kl_div.
# Our optimized version uses a fused approach that avoids computing log separately
# by directly applying log_softmax to the predictions and computing KL divergence efficiently.