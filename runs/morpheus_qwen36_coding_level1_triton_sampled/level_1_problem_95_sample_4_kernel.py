import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_loss_kernel(
    predictions_ptr,
    targets_ptr,
    out_losses_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Load logits for the current sample
    logits = tl.load(predictions_ptr + batch_idx * num_classes + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(logits, axis=0)
    
    # Shift logits and compute exp
    shifted_logits = logits - max_val
    exp_shifted = tl.exp(shifted_logits)
    
    # Sum exp and compute log-sum-exp
    sum_exp = tl.sum(exp_shifted, axis=0)
    lse = tl.log(sum_exp)
    
    # Load target and correct logit
    target = tl.load(targets_ptr + batch_idx)
    correct_logit = tl.load(predictions_ptr + batch_idx * num_classes + target)
    
    # Compute loss for this sample: logsumexp - correct_logit
    sample_loss = lse - correct_logit
    
    # Store result
    tl.store(out_losses_ptr + batch_idx, sample_loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    
    out_losses = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
    
    # Set block size to num_classes (power of 2 for optimal performance)
    BLOCK_SIZE = num_classes
    
    grid = (batch_size,)
    cross_entropy_loss_kernel[grid](
        predictions, targets, out_losses,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return torch.mean(out_losses)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)