import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    tgt = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute difference
    diff = pred - tgt
    abs_diff = tl.abs(diff)
    
    # Smooth L1 loss: 
    # if |x| < beta: 0.5 * x^2 / beta
    # else: |x| - 0.5 * beta
    loss = tl.where(
        abs_diff < beta,
        0.5 * diff * diff / beta,
        abs_diff - 0.5 * beta
    )

    # Store result
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Computes Smooth L1 (Huber) Loss using Triton kernel.
    
    Args:
        predictions: Predicted values tensor
        targets: Ground truth values tensor
        beta: Threshold for quadratic vs linear loss
        
    Returns:
        Mean of the Smooth L1 loss over all elements
    """
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # Prepare output tensor (same shape as inputs)
    output = torch.empty_like(predictions)
    
    # Set block size for kernel
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    smooth_l1_loss_kernel[grid](
        predictions, 
        targets, 
        output, 
        n_elements, 
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss (as per PyTorch's default behavior)
    return output.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using custom Triton kernel.
    
    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Default beta value for Smooth L1 loss
        self.beta = 1.0

    def forward(self, predictions, targets):
        # Use custom Triton implementation of Smooth L1 loss
        return triton_smooth_l1_loss(predictions, targets, beta=self.beta)