import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    predictions_ptr, 
    targets_ptr, 
    loss_ptr, 
    stride_row, 
    num_classes, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel to compute the per-sample cross entropy loss.
    
    Formula: loss = log(sum(exp(logits))) - logit[target]
    To ensure numerical stability, we use the Log-Sum-Exp trick:
    LSE = max(logits) + log(sum(exp(logits - max(logits))))
    """
    # Each program processes one sample in the batch
    row_idx = tl.program_id(0)
    
    # Pointer to the start of the current row in the predictions tensor
    row_ptr = predictions_ptr + row_idx * stride_row
    
    # Load the logits for the current sample
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    logits = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute Log-Sum-Exp for numerical stability
    max_val = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - max_val), axis=0)
    lse = max_val + tl.log(sum_exp)
    
    # Load the target class index for this sample
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Load the logit corresponding to the target class
    target_logit = tl.load(row_ptr + target_idx)
    
    # Cross Entropy Loss for a single sample
    loss = lse - target_logit
    
    # Store the per-sample loss in the output tensor
    tl.store(loss_ptr + row_idx, loss)

def triton_cross_entropy(predictions, targets):
    """
    Wrapper function to launch the Triton cross entropy kernel.
    """
    # Ensure inputs are on GPU and contiguous
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    N, C = predictions.shape
    
    # Output tensor for per-sample loss
    loss = torch.empty(N, device=predictions.device, dtype=predictions.dtype)
    
    # BLOCK_SIZE must be a power of 2 and at least as large as the number of classes
    # For num_classes = 4096, this will be 4096.
    BLOCK_SIZE = 1 << (C - 1).bit_length()
    
    # Grid: one program per sample in the batch
    grid = (N,)
    
    cross_entropy_kernel[grid](
        predictions, 
        targets, 
        loss, 
        predictions.stride(0), 
        C, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # PyTorch's cross_entropy returns the mean by default
    return loss.mean()

class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.nn.functional.cross_entropy with our Triton implementation
        return triton_cross_entropy(predictions, targets)