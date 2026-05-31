import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    predictions_ptr, 
    targets_ptr, 
    out_ptr, 
    num_classes, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one sample in the batch
    pid = tl.program_id(0)
    
    # Pointer to the start of the predictions for the current sample
    row_ptr = predictions_ptr + pid * num_classes
    
    # Load the predictions for the current sample
    # We use a block size that is a power of 2 and >= num_classes
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    x = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Log-Sum-Exp trick for numerical stability:
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    max_val = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - max_val), axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Load the target class index for the current sample
    target_idx = tl.load(targets_ptr + pid)
    
    # Load the prediction value for the target class
    target_val = tl.load(row_ptr + target_idx)
    
    # Cross Entropy Loss = -log(exp(target_val) / sum(exp(x_i)))
    # = log(sum(exp(x_i))) - target_val
    loss = log_sum_exp - target_val
    
    # Store the loss for the current sample
    tl.store(out_ptr + pid, loss)

def triton_cross_entropy(predictions, targets):
    """
    Triton implementation of Cross Entropy Loss.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    
    # Output tensor to store loss per sample
    out = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
    
    # BLOCK_SIZE must be a power of 2 and >= num_classes
    # For num_classes = 4096, 4096 is a power of 2.
    BLOCK_SIZE = triton.next_power_of_2(num_classes)
    
    # Grid is one program per batch element
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions, 
        targets, 
        out, 
        num_classes, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean loss across the batch to match torch.nn.functional.cross_entropy
    return out.mean()

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using a custom Triton kernel for speedup.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.nn.functional.cross_entropy with our Triton implementation
        return triton_cross_entropy(predictions, targets)