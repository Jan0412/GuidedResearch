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
    grad_logits_ptr,  # [batch_size, num_classes]
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # We'll compute the loss and gradient in a fused manner
    # First pass: compute max for numerical stability and softmax denominator
    # Second pass: compute loss and gradients
    
    # Process one row (sample) per program
    batch_id = tl.program_id(0)
    
    # Offset to the current row in logits
    logits_row_start = batch_id * num_classes
    
    # Load target for this sample
    target_idx = tl.load(targets_ptr + batch_id)
    
    # Compute max for numerical stability (online max)
    max_val = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        row_offsets = logits_row_start + offsets
        vals = tl.load(logits_ptr + row_offsets, mask=mask, other=-float('inf'))
        row_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, row_max)
    
    # Compute softmax denominator (sum of exp(x - max))
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        row_offsets = logits_row_start + offsets
        vals = tl.load(logits_ptr + row_offsets, mask=mask, other=-float('inf'))
        shifted = vals - max_val
        exp_vals = tl.exp(shifted)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Compute log(sum_exp) = log_softmax denominator
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Compute the loss for this sample: -logits[target] + log(sum_exp)
    # Load the target logit
    target_offset = logits_row_start + target_idx
    target_logit = tl.load(logits_ptr + target_offset)
    
    # Sample loss: -target_logit + log_sum_exp
    sample_loss = -target_logit + log_sum_exp
    
    # Store the loss (we'll accumulate and average in host code)
    tl.store(loss_ptr, sample_loss)
    
    # Compute gradients: softmax(logits) - indicator(target)
    # First compute softmax probabilities
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        row_offsets = logits_row_start + offsets
        vals = tl.load(logits_ptr + row_offsets, mask=mask, other=-float('inf'))
        shifted = vals - max_val
        exp_vals = tl.exp(shifted)
        probs = exp_vals / sum_exp
        
        # Gradient: probs - (1 if offset == target else 0)
        target_mask = (offsets == target_idx)
        grad = tl.where(target_mask, probs - 1.0, probs)
        
        # Store gradients
        tl.store(grad_logits_ptr + row_offsets, grad, mask=mask)


@triton.jit
def reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Simple kernel to sum up the per-sample losses
    # Each block sums a portion of the array
    offset = tl.program_id(0) * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(data, axis=0)
    tl.store(output_ptr + tl.program_id(0), sum_val)


def triton_cross_entropy(logits, targets):
    """
    Compute cross entropy loss using Triton kernels.
    
    Args:
        logits: [batch_size, num_classes] tensor of raw predictions
        targets: [batch_size] tensor of class indices
    
    Returns:
        Scalar cross entropy loss
    """
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert logits.dim() == 2, "Logits must be 2D: [batch_size, num_classes]"
    assert targets.dim() == 1, "Targets must be 1D: [batch_size]"
    
    batch_size, num_classes = logits.shape
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    # Allocate output for loss (we'll accumulate per-sample losses first)
    per_sample_loss = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    grad_logits = torch.empty_like(logits)
    
    # Launch the main cross entropy kernel - one block per batch sample
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    # We'll use a temporary buffer to store per-sample losses
    cross_entropy_kernel[grid](
        logits, targets, per_sample_loss, grad_logits,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum up per-sample losses and divide by batch size
    total_loss = per_sample_loss.sum() / batch_size
    
    # Store gradients for autograd (we'll create a custom autograd function)
    # For simplicity in this implementation, we'll use a custom autograd approach
    return total_loss


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        batch_size, num_classes = logits.shape
        logits = logits.contiguous()
        targets = targets.contiguous()
        
        per_sample_loss = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
        grad_logits = torch.empty_like(logits)
        
        BLOCK_SIZE = 256
        grid = (batch_size,)
        
        # Launch kernel - store per-sample losses in a temporary buffer for gradient computation
        cross_entropy_kernel[grid](
            logits, targets, per_sample_loss, grad_logits,
            batch_size, num_classes,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(grad_logits, targets)
        
        total_loss = per_sample_loss.sum() / batch_size
        return total_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_logits, targets = ctx.saved_tensors
        batch_size = grad_logits.shape[0]
        
        # The gradient of cross entropy is (softmax - one_hot) / batch_size
        # We already computed softmax - one_hot in the forward pass (stored in grad_logits)
        # Scale by grad_output (which is 1.0 for loss) and divide by batch_size
        return grad_logits * grad_output / batch_size, None


def triton_cross_entropy_autograd(logits, targets):
    return CrossEntropyFunction.apply(logits, targets)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for cross entropy loss.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy_autograd(predictions, targets)