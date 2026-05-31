import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

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
    
    # Calculate base offset for this batch and sequence
    base_offset = batch_idx * seq_len + seq_idx * seq_len
    
    # Each program handles one element in the sequence
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load prediction and target values
    pred = tl.load(predictions_ptr + base_offset + offsets, mask=mask, other=0.0)
    target = tl.load(targets_ptr + base_offset + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence contribution: target * log(target / prediction)
    # Handle numerical stability
    eps = 1e-8
    pred = tl.maximum(pred, eps)
    target = tl.maximum(target, eps)
    
    # Compute log(target / prediction) = log(target) - log(prediction)
    log_target = tl.log(target)
    log_pred = tl.log(pred)
    log_ratio = log_target - log_pred
    
    # Compute kl_div = target * log(target / prediction)
    kl_contribution = target * log_ratio
    
    # Store the result for this element
    kl_result = tl.sum(kl_contribution)
    
    # Store the result for this batch
    output_offset = batch_idx
    tl.atomic_add(output_ptr + output_offset, kl_result)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton kernel for KL divergence computation
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    batch_size, seq_len = predictions.shape
    
    # Prepare output tensor (sum per batch)
    out = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        1,           # sequence dimension (we process entire sequences)
        (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # element processing dimension
    )
    
    # Launch kernel
    kl_div_kernel[grid](
        predictions,
        targets,
        out,
        predictions.numel(),
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean across batches
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for KL divergence computation
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our Triton-based KL divergence instead of PyTorch's implementation
        return triton_kl_div(predictions, targets)