import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute 1 - predictions * targets
    temp = 1.0 - predictions * targets
    
    # Apply clamp(max=0) which is equivalent to max(0, -temp)
    clamped = tl.maximum(0.0, -temp)
    
    # Store result
    tl.store(output_ptr + offsets, clamped, mask=mask)

@triton.jit
def mean_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sum reduction within block
    block_sum = tl.sum(input_vals, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), block_sum, mask=tl.program_id(0) < (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute hinge loss using Triton kernel
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for intermediate results
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Allocate output tensor for intermediate values
    intermediate = torch.empty_like(predictions)
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel for computing hinge loss per element
    hinge_loss_kernel[grid](predictions, targets, intermediate, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Now compute mean using another kernel approach
    # For better efficiency, we'll use a more optimized reduction approach
    if n_elements <= 1024:
        # Direct mean computation for small tensors
        return torch.mean(intermediate)
    else:
        # Use a two-phase approach for large tensors
        # First phase: reduce to smaller chunks
        chunk_size = 1024
        num_chunks = (n_elements + chunk_size - 1) // chunk_size
        
        # Allocate temporary storage for partial sums
        partial_sums = torch.zeros(num_chunks, device=predictions.device, dtype=torch.float32)
        
        # Launch reduction kernel for partial sums
        reduction_grid = lambda meta: (num_chunks,)
        mean_reduction_kernel[reduction_grid](
            intermediate, 
            partial_sums, 
            n_elements, 
            BLOCK_SIZE=chunk_size
        )
        
        # Final reduction to scalar mean
        total_sum = torch.sum(partial_sums)
        return total_sum / n_elements

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for hinge loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)