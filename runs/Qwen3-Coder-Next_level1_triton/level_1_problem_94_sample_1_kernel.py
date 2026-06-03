import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Custom Triton kernel for computing MSE loss efficiently."""
    # Compute the global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared error
    diff = predictions - targets
    squared_error = diff * diff
    
    # Accumulate for reduction (we'll do the final mean in a separate step)
    # For now, just store the squared errors
    tl.store(output_ptr + offsets, squared_error, mask=mask)


@triton.jit
def reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to compute sum reduction."""
    # Use a simple approach: first compute partial sums in parallel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and sum
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    # Store result
    tl.store(output_ptr + pid, sum_val)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss using Triton kernels.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 256
    
    # First kernel: compute squared errors and store in intermediate buffer
    intermediate = torch.empty_like(predictions)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mse_loss_kernel[grid](predictions, targets, intermediate, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: reduce to get sum, but for simplicity and since PyTorch's mean is efficient,
    # we'll use PyTorch for the final reduction step if the tensor is small enough
    # Otherwise, use a two-level reduction
    
    # For simplicity and robustness, use PyTorch's mean for the final reduction
    # This handles both small and large tensors efficiently
    # But to be more Triton-optimized, let's do a proper reduction
    
    # For very large tensors, we can do a two-level reduction
    if n_elements <= BLOCK_SIZE * 1024:
        # For smaller tensors, just use PyTorch mean on the intermediate result
        # This avoids overhead of complex reduction logic
        squared_errors = intermediate.view(-1)
        return torch.mean(squared_errors)
    else:
        # For very large tensors, implement a more sophisticated reduction
        # First, compute partial sums
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        partial_sums = torch.empty(num_blocks, device=predictions.device, dtype=predictions.dtype)
        
        grid = lambda meta: ((num_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        # Actually, we need to compute num_blocks first
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        partial_sums = torch.empty(num_blocks, device=predictions.device, dtype=predictions.dtype)
        
        reduce_sum_kernel[(num_blocks,)](intermediate, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        # Then sum the partial sums
        return torch.sum(partial_sums) / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)