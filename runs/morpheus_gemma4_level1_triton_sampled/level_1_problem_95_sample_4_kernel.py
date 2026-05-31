import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    predictions_ptr, targets_ptr, loss_ptr,
    batch_size, num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Pointers for the current row of logits
    row_start_ptr = predictions_ptr + row_idx * num_classes
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes

    # Load logits for the row
    # Use -inf for the mask to ensure they don't affect the max or sum of exps
    logits = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))

    # Log-Sum-Exp trick for numerical stability
    # Find max logit in the row
    max_logit = tl.max(logits, axis=0)
    
    # Compute sum(exp(logits - max_logit))
    shifted_logits = logits - max_logit
    exp_logits = tl.exp(shifted_logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # Log-Sum-Exp = max_logit + log(sum_exp)
    lse = max_logit + tl.log(sum_exp)

    # Load the target class index for this row
    target_id = tl.load(targets_ptr + row_idx)
    # Load the logit corresponding to the target class
    target_logit = tl.load(row_start_ptr + target_id)

    # Cross entropy loss for this row: LSE - target_logit
    row_loss = lse - target_logit

    # Atomically add the row loss to the global sum
    tl.atomic_add(loss_ptr, row_loss)

def triton_cross_entropy(predictions, targets):
    # Ensure inputs are on CUDA and contiguous
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    batch_size, num_classes = predictions.shape
    # Initialize global loss as a 0-dim tensor on the correct device
    loss = torch.zeros((), device=predictions.device, dtype=torch.float32)

    # BLOCK_SIZE must be a power of 2 and cover all classes for the reduction
    BLOCK_SIZE = triton.next_power_of_2(num_classes)
    
    # Grid: one program per row in the batch
    grid = (batch_size,)

    cross_entropy_kernel[grid](
        predictions, targets, loss,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Average the loss over the batch
    return loss / batch_size

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using a custom Triton kernel for improved performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)