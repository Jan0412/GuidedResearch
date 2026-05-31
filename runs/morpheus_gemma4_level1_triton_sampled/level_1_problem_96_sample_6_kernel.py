import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_kernel(
    x_ptr,  # Pointer to predictions
    y_ptr,  # Pointer to targets
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute absolute difference
    diff = tl.abs(x - y)

    # Smooth L1 formula (beta = 1.0)
    # loss = 0.5 * diff^2 if diff < 1 else diff - 0.5
    loss = tl.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)

    # Store the result
    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton wrapper for smooth_l1_loss with reduction='mean'.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for Triton
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    out = torch.empty_like(predictions)

    # Block size for processing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel to compute element-wise loss
    smooth_l1_kernel[grid](
        predictions, 
        targets, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Perform reduction (mean) using PyTorch's optimized mean operator
    return out.mean()


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks,
    optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.nn.functional.smooth_l1_loss with Triton implementation
        return triton_smooth_l1_loss(predictions, targets)