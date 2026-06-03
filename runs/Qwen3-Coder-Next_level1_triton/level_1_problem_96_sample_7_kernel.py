import torch
import torch.nn as nn
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
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute difference
    diff = predictions - targets
    abs_diff = tl.abs(diff)

    # Smooth L1 loss formula:
    # if |x| < beta: 0.5 * x^2 / beta
    # else: |x| - 0.5 * beta
    condition = abs_diff < beta
    loss = tl.where(
        condition,
        0.5 * (diff * diff) / beta,
        abs_diff - 0.5 * beta
    )

    # Store the result
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    This function wraps the Triton kernel call for Smooth L1 (Huber) Loss. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
      4. Returns the mean of the computed loss (as per PyTorch's smooth_l1_loss default behavior).
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Prepare output tensor (same shape as inputs)
    out = torch.empty_like(predictions)

    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    smooth_l1_loss_kernel[grid](
        predictions, targets, out, n_elements, beta=beta, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean as per PyTorch's default reduction='mean' behavior
    return out.mean()


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks using a custom Triton kernel.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton-based smooth L1 loss with default beta=1.0 (same as PyTorch)
        return triton_smooth_l1_loss(predictions, targets, beta=1.0)