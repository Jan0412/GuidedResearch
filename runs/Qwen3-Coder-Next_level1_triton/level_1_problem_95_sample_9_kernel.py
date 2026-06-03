import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,  # [batch_size, num_classes]
    targets_ptr,  # [batch_size]
    loss_ptr,  # scalar output
    batch_size: tl.constexpr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Load target for this batch
    target_idx = tl.load(targets_ptr + batch_idx)
    
    # Compute log-softmax and loss for this batch
    # Start by finding max for numerical stability
    max_val = tl.minimum(tl.float32(-1e30))
    sum_exp = tl.float32(0.0)
    
    # Process in blocks to compute max and sum of exp
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(logits_ptr + batch_idx * num_classes + offsets, mask=mask, other=-1e30)
        
        # Update max
        block_max = tl.max(logits)
        max_val = tl.maximum(max_val, block_max)
    
    # Now compute sum of exp(logits - max_val)
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(logits_ptr + batch_idx * num_classes + offsets, mask=mask, other=-1e30)
        
        # Compute exp(logits - max_val)
        exp_logits = tl.exp(logits - max_val)
        sum_exp += tl.sum(exp_logits, mask=mask)
    
    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Compute loss for target class: -logits[target] + log_sum_exp
    target_offset = batch_idx * num_classes + target_idx
    target_logit = tl.load(logits_ptr + target_offset)
    
    # Negative log likelihood for this sample
    loss = -target_logit + log_sum_exp
    
    # Store loss
    tl.store(loss_ptr + batch_idx, loss)


@triton.jit
def cross_entropy_backward_kernel(
    logits_ptr,  # [batch_size, num_classes]
    targets_ptr,  # [batch_size]
    grad_output_ptr,  # [batch_size] - gradient of loss w.r.t. each sample
    grad_logits_ptr,  # [batch_size, num_classes]
    batch_size: tl.constexpr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Load target for this batch
    target_idx = tl.load(targets_ptr + batch_idx)
    
    # Compute softmax for this batch
    # First find max for numerical stability
    max_val = tl.minimum(tl.float32(-1e30))
    
    # Process in blocks to compute max
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(logits_ptr + batch_idx * num_classes + offsets, mask=mask, other=-1e30)
        
        # Update max
        block_max = tl.max(logits)
        max_val = tl.maximum(max_val, block_max)
    
    # Now compute softmax probabilities
    sum_exp = tl.float32(0.0)
    
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load logits
        logits = tl.load(logits_ptr + batch_idx * num_classes + offsets, mask=mask, other=-1e30)
        
        # Compute exp(logits - max_val)
        exp_logits = tl.exp(logits - max_val)
        
        # Store for later use
        tl.store(grad_logits_ptr + batch_idx * num_classes + offsets, exp_logits, mask=mask)
        
        sum_exp += tl.sum(exp_logits, mask=mask)
    
    # Compute softmax probabilities by dividing by sum
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load exp logits
        exp_logits = tl.load(grad_logits_ptr + batch_idx * num_classes + offsets, mask=mask)
        
        # Compute softmax = exp(logits - max) / sum_exp
        softmax = exp_logits / sum_exp
        
        # Store softmax probabilities
        tl.store(grad_logits_ptr + batch_idx * num_classes + offsets, softmax, mask=mask)
    
    # Compute gradients: dL/dlogits[i] = softmax[i] - (i == target)
    # Scale by grad_output[batch_idx]
    grad_scale = tl.load(grad_output_ptr + batch_idx)
    
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load softmax probabilities
        softmax = tl.load(grad_logits_ptr + batch_idx * num_classes + offsets, mask=mask)
        
        # Create indicator mask for target
        is_target = (offsets == target_idx)
        
        # Gradient: (softmax - indicator) * grad_scale
        grad = (softmax - tl.where(is_target, tl.float32(1.0), tl.float32(0.0))) * grad_scale
        
        # Store gradient
        tl.store(grad_logits_ptr + batch_idx * num_classes + offsets, grad, mask=mask)


def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    Compute cross entropy loss using Triton kernel.
    
    Args:
        logits: [batch_size, num_classes] input logits
        targets: [batch_size] target class indices
    
    Returns:
        loss: scalar tensor with mean cross entropy loss
    """
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = logits.shape
    
    # Allocate output for per-sample losses
    per_sample_loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    
    # Launch kernel with one block per batch element
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        logits, targets, per_sample_loss,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return mean loss
    return per_sample_loss.mean()


def triton_cross_entropy_backward(grad_output: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
    """
    Compute backward pass for cross entropy loss using Triton kernel.
    
    Args:
        grad_output: gradient of loss w.r.t. output (typically 1.0/batch_size for mean loss)
        logits: [batch_size, num_classes] input logits
        targets: [batch_size] target class indices
    
    Returns:
        grad_logits: gradient of loss w.r.t. logits
    """
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    grad_output = grad_output.contiguous()
    
    batch_size, num_classes = logits.shape
    
    # Allocate gradient tensor (will be overwritten with gradients)
    grad_logits = torch.empty_like(logits)
    
    # Launch kernel with one block per batch element
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    cross_entropy_backward_kernel[grid](
        logits, targets, grad_output, grad_logits,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad_logits


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        return triton_cross_entropy(logits, targets)
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        return triton_cross_entropy_backward(grad_output, logits, targets), None


def cross_entropy_triton(logits: torch.Tensor, targets: torch.Tensor):
    """
    Functional interface for Triton-based cross entropy loss.
    """
    return CrossEntropyFunction.apply(logits, targets)


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.
    Uses optimized Triton kernels for computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our Triton-based cross entropy implementation
        return cross_entropy_triton(predictions, targets)