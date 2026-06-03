import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X,  # pointer to input logits
    Y,  # pointer to output log-softmax
    N_COLS,  # number of columns (classes)
    BLOCK_SIZE: tl.constexpr
):
    # Get row index
    row_idx = tl.program_id(0)
    
    # Compute offsets for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS
    
    # Load logits for this row
    x_ptr = X + row_idx * N_COLS
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exponentials of shifted values
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    
    # Compute sum of exponentials
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute log-sum-exp
    lse = x_max + tl.log(x_sum)
    
    # Compute log-softmax: x - lse
    log_softmax = x - lse
    
    # Store result
    y_ptr = Y + row_idx * N_COLS
    tl.store(y_ptr + offsets, log_softmax, mask=mask)


@triton.jit
def cross_entropy_loss_kernel(
    log_probs_ptr,  # pointer to log-softmax output
    targets_ptr,    # pointer to target indices
    loss_ptr,       # pointer to loss output
    n_rows,         # number of rows (batch size)
    n_cols,         # number of columns (classes)
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row (one sample)
    row_idx = tl.program_id(0)
    
    # Get target index for this row
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Compute offset to the target position in log_probs
    target_offset = row_idx * n_cols + target_idx
    
    # Load the log probability at target position
    log_prob = tl.load(log_probs_ptr + target_offset)
    
    # Compute negative log probability (cross entropy for this sample)
    loss = -log_prob
    
    # Store result
    tl.store(loss_ptr + row_idx, loss)


class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        batch_size, num_classes = predictions.shape
        
        # Ensure inputs are contiguous
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Allocate memory for log-softmax output
        log_probs = torch.empty_like(predictions)
        
        # Launch log-softmax kernel
        BLOCK_SIZE = 128
        grid_log_softmax = (batch_size,)
        log_softmax_kernel[grid_log_softmax](
            predictions, log_probs, num_classes,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Allocate memory for per-sample losses
        per_sample_losses = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
        
        # Launch cross entropy loss kernel
        grid_loss = (batch_size,)
        cross_entropy_loss_kernel[grid_loss](
            log_probs, targets, per_sample_losses,
            batch_size, num_classes,
            BLOCK_SIZE=128
        )
        
        # Compute mean loss
        loss = per_sample_losses.mean()
        
        # Save for backward pass
        ctx.save_for_backward(predictions, targets, log_probs)
        ctx.num_classes = num_classes
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets, log_probs = ctx.saved_tensors
        num_classes = ctx.num_classes
        
        batch_size = predictions.shape[0]
        
        # Compute gradients:
        # dL/d(log_probs) = (1/batch_size) * exp(log_probs) - (1/batch_size) * one_hot(targets)
        # Since we have log_probs, we compute probs = exp(log_probs)
        # Then: grad = probs - one_hot(targets)
        # Finally: multiply by grad_output
        
        # Compute softmax probabilities from log_probs
        probs = torch.exp(log_probs)
        
        # Create one-hot encoding of targets
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Compute gradient: (probs - one_hot) * grad_output / batch_size
        # Since grad_output is scalar (1.0 for mean loss), we just need (probs - one_hot) / batch_size
        grad_logits = (probs - one_hot) / batch_size * grad_output
        
        return grad_logits, None


def triton_cross_entropy(predictions, targets):
    return TritonCrossEntropyFunction.apply(predictions, targets)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for cross entropy loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)