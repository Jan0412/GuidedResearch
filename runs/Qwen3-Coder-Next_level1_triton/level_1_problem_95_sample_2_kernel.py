import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    X_ptr,  # [batch_size, num_classes] logit tensor
    Y_ptr,  # [batch_size] target indices
    loss_ptr,  # [batch_size] output loss per sample
    log_sum_exp_ptr,  # [batch_size] log sum exp of logits per sample (for gradient computation)
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample (one row of the input)
    sample_idx = tl.program_id(0)
    
    # Compute base pointers for this sample
    x_offset = sample_idx * num_classes
    
    # Store log sum exp for this sample (initialized to -inf)
    log_sum_exp = -float('inf')
    
    # Compute max logit for numerical stability
    max_logit = -float('inf')
    
    # Loop over classes in blocks
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(X_ptr + x_offset + offsets, mask=mask, other=-float('inf'))
        
        # Update max logit
        max_logit = tl.maximum(max_logit, tl.max(logits))
    
    # Now compute log sum exp using the max for stability
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(X_ptr + x_offset + offsets, mask=mask, other=-float('inf'))
        
        # Compute exp(logit - max) for stability
        exp_logits = tl.exp(logits - max_logit)
        
        # Accumulate sum
        sum_exp = tl.sum(exp_logits * mask)
        log_sum_exp = log_sum_exp + tl.log(sum_exp)
    
    # Add max back (log sum exp = max + log(sum(exp(logits - max))))
    log_sum_exp = max_logit + log_sum_exp
    
    # Load the target index
    target_idx = tl.load(Y_ptr + sample_idx)
    
    # Compute loss: -logit[target] + log_sum_exp
    # Note: we only need the logit at target_idx, so load just that element
    target_logit = tl.load(X_ptr + x_offset + target_idx)
    
    # Cross entropy for this sample
    loss = -target_logit + log_sum_exp
    
    # Store results
    tl.store(loss_ptr + sample_idx, loss)
    tl.store(log_sum_exp_ptr + sample_idx, log_sum_exp)


@triton.jit
def cross_entropy_backward_kernel(
    grad_output_ptr,  # [batch_size] gradient from loss
    X_ptr,  # [batch_size, num_classes] logits
    Y_ptr,  # [batch_size] targets
    log_sum_exp_ptr,  # [batch_size] precomputed log sum exp
    dX_ptr,  # [batch_size, num_classes] gradient w.r.t. logits
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample
    sample_idx = tl.program_id(0)
    
    # Compute base pointers
    x_offset = sample_idx * num_classes
    dx_offset = sample_idx * num_classes
    
    # Load precomputed log_sum_exp for this sample
    log_sum_exp = tl.load(log_sum_exp_ptr + sample_idx)
    
    # Load gradient from next layer
    grad_out = tl.load(grad_output_ptr + sample_idx)
    
    # Compute softmax denominator: exp(log_sum_exp) = sum(exp(logits))
    # But we don't need to compute it explicitly; we can use exp(logits - log_sum_exp) = softmax
    # Instead, we compute exp(logits - log_sum_exp) = exp(logits) / sum(exp(logits))
    
    # For numerical stability, compute softmax using exp(logits - max) and divide by sum(exp(logits - max))
    # But we already have log_sum_exp, so: softmax_i = exp(logits_i - log_sum_exp)
    
    # Compute softmax probabilities for all classes
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0)
        
        # Compute softmax: exp(logits - log_sum_exp)
        softmax = tl.exp(logits - log_sum_exp)
        
        # Load target index
        target_idx = tl.load(Y_ptr + sample_idx)
        
        # For the target class, subtract 1 from softmax
        # We need to check if each offset is the target index
        target_mask = (offsets == target_idx)
        softmax = softmax - target_mask.to(tl.float32)
        
        # Multiply by grad_out and store
        dX = grad_out * softmax
        tl.store(dX_ptr + dx_offset + offsets, dX, mask=mask)


def triton_cross_entropy(predictions, targets):
    """
    Compute cross entropy loss using Triton kernels.
    
    Args:
        predictions: [batch_size, num_classes] logits
        targets: [batch_size] target class indices
    
    Returns:
        loss: scalar cross entropy loss (mean over batch)
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    loss = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    log_sum_exp = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    BLOCK_SIZE = 256
    
    # Launch kernel for forward pass
    grid = (batch_size,)
    cross_entropy_kernel[grid](
        predictions, targets, loss, log_sum_exp,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return loss.mean()


def triton_cross_entropy_backward(grad_output, predictions, targets, log_sum_exp):
    """
    Compute gradient of cross entropy loss w.r.t. predictions.
    
    Args:
        grad_output: gradient from next layer (scalar or [batch_size])
        predictions: [batch_size, num_classes] logits
        targets: [batch_size] target class indices
        log_sum_exp: [batch_size] precomputed log sum exp
    
    Returns:
        d_predictions: gradient w.r.t. predictions
    """
    batch_size, num_classes = predictions.shape
    d_predictions = torch.empty_like(predictions)
    
    BLOCK_SIZE = 256
    
    # Handle grad_output being a scalar
    if grad_output.numel() == 1:
        grad_output_expanded = grad_output.expand(batch_size).contiguous()
    else:
        grad_output_expanded = grad_output.contiguous()
    
    # Launch kernel for backward pass
    grid = (batch_size,)
    cross_entropy_backward_kernel[grid](
        grad_output_expanded, predictions, targets, log_sum_exp,
        d_predictions,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return d_predictions


# Custom autograd function for cross entropy
class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        # Save for backward
        ctx.save_for_backward(predictions, targets)
        
        # Compute log_sum_exp for backward pass
        batch_size, num_classes = predictions.shape
        log_sum_exp = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
        
        BLOCK_SIZE = 256
        grid = (batch_size,)
        cross_entropy_kernel[grid](
            predictions, targets, None, log_sum_exp,
            batch_size, num_classes,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Compute loss (mean over batch)
        # For efficiency, we can compute mean directly in kernel, but for simplicity use PyTorch
        loss_tensor = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
        cross_entropy_kernel[grid](
            predictions, targets, loss_tensor, log_sum_exp,
            batch_size, num_classes,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Store log_sum_exp for backward
        ctx.log_sum_exp = log_sum_exp
        
        return loss_tensor.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        log_sum_exp = ctx.log_sum_exp
        
        # Compute gradient
        grad_predictions = triton_cross_entropy_backward(grad_output, predictions, targets, log_sum_exp)
        
        # targets don't have gradients
        return grad_predictions, None


def cross_entropy(predictions, targets):
    """
    Wrapper function for cross entropy using our custom autograd function.
    """
    return CrossEntropyFunction.apply(predictions, targets)


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using custom Triton kernels.
    
    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our custom cross entropy implementation
        return cross_entropy(predictions, targets)