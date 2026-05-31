import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    abs_diff = tl.abs(diff)
    
    # Compute smooth L1 loss
    # if abs_diff < beta: 
    #     loss = 0.5 * (diff ** 2) / beta
    # else:
    #     loss = abs_diff - 0.5 * beta
    
    condition = abs_diff < beta
    squared_loss = 0.5 * (diff * diff) / beta
    linear_loss = abs_diff - 0.5 * beta
    loss = tl.where(condition, squared_loss, linear_loss)
    
    tl.store(out_ptr + offsets, loss, mask=mask)

@triton.jit
def smooth_l1_loss_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce sum
    reduced = tl.sum(input_vals, axis=0)
    
    # Store the reduced value
    tl.store(output_ptr, reduced, mask=mask)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Custom Triton implementation of Smooth L1 Loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # First kernel: compute individual losses
    loss_buffer = torch.empty(n_elements, dtype=torch.float32, device=predictions.device)
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        loss_buffer,
        n_elements,
        beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: reduce to scalar
    output = torch.empty(1, dtype=torch.float32, device=predictions.device)
    
    # For reduction, we can use a simple approach with a single block for small tensors
    # Or better yet, use a proper reduction kernel
    if n_elements <= 1024:
        smooth_l1_loss_reduce_kernel[1](
            loss_buffer,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # For large tensors, we need to properly reduce
        # This is a simplified version - in practice, you'd want more sophisticated reduction
        output.fill_(loss_buffer.sum().item())
    
    return output

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Smooth L1 Loss computation
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)