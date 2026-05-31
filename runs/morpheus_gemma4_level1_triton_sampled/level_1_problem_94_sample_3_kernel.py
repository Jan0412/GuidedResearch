import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_reduction_kernel(
    x_ptr,  # Pointer to predictions
    y_ptr,  # Pointer to targets
    out_ptr,  # Pointer to output partial sums
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference: (x - y)^2
    diff = x - y
    sq = diff * diff
    
    # Local reduction within the block
    # We use tl.sum to get the sum of the squared differences in this block
    block_sum = tl.sum(sq, axis=0)
    
    # Store the partial sum in the output buffer
    tl.store(out_ptr + pid, block_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Fuses the subtraction, squaring, and partial sum into a single Triton kernel,
    then performs a final reduction using PyTorch.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Standard block size for FP32 reductions
    
    # Calculate the number of blocks needed
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    
    # Buffer to store partial sums from each block
    partial_sums = torch.empty(grid_size, device=predictions.device, dtype=torch.float32)
    
    # Launch the Triton kernel
    mse_reduction_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction: sum all partial sums and divide by total elements
    # torch.sum is highly optimized for these sizes
    total_sum = torch.sum(partial_sums)
    return total_sum / n_elements


class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with the fused Triton implementation
        return triton_mse(predictions, targets)