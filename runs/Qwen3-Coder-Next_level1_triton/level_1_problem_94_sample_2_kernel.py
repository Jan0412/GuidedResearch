import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Use reduction to compute sum of squared differences
    sum_acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tgt = tl.load(targets_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute squared difference and accumulate
        diff = pred - tgt
        sum_acc += tl.sum(diff * diff)
    
    # Write the sum to output_ptr (we'll handle the mean division in the wrapper)
    tl.store(output_ptr, sum_acc)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes Mean Squared Error loss using Triton kernel.
    
    Parameters:
        predictions: Predicted tensor
        targets: Target tensor
        
    Returns:
        Mean squared error as a scalar tensor
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Create output tensor to store the sum (will be divided by n_elements later)
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    
    # Launch the kernel
    mse_loss_kernel[(1,)](
        predictions, 
        targets, 
        output, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute the mean by dividing by number of elements
    return output[0] / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using Triton kernel.
    
    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)