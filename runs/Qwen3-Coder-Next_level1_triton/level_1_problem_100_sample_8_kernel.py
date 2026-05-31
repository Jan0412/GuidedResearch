import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    loss_ptr,         # Pointer to output loss values
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - prediction * target)
    prod = predictions * targets
    loss = tl.maximum(0.0, 1.0 - prod)
    
    # Store result
    tl.store(loss_ptr + offsets, loss, mask=mask)


@triton.jit
def mean_kernel(
    input_ptr,      # Pointer to input
    output_ptr,     # Pointer to output (single scalar)
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial sum
    sum_val = tl.sum(x, axis=0)
    
    # Accumulate using atomic add for correctness (though for single output we can use shared memory)
    # Since this is for a single output, we'll use a simple approach with atomic_add
    tl.atomic_add(output_ptr, sum_val)


@triton.jit
def finalize_mean_kernel(
    sum_ptr,        # Pointer to accumulated sum
    output_ptr,     # Pointer to final mean output
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel computes the final mean by dividing the sum by n_elements
    # Since there's only one element in sum_ptr, we just do the division
    if tl.program_id(0) == 0:
        sum_val = tl.load(sum_ptr)
        mean = sum_val / n_elements
        tl.store(output_ptr, mean)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes hinge loss using Triton kernels.
    
    Parameters:
        predictions: Tensor of shape (batch_size,)
        targets: Tensor of shape (batch_size,)
        
    Returns:
        Scalar tensor containing the mean hinge loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Shapes must match."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Create output tensor for loss values
    loss_values = torch.empty_like(predictions)
    
    # First kernel: compute hinge loss for each element
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hinge_loss_kernel[grid](predictions, targets, loss_values, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute sum using atomic operations
    total_sum = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mean_kernel[grid](loss_values, total_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final kernel: compute mean
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    finalize_mean_kernel[1](total_sum, output, n_elements, BLOCK_SIZE=1)
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)