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
    # Each program instance accumulates a partial sum
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Accumulate for reduction (this will be summed across blocks)
    tl.atomic_add(output_ptr, squared_diff, mask=mask)


@triton.jit
def final_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the final mean by dividing the sum by n_elements
    # This kernel is called with a single block
    if tl.program_id(0) == 0:
        total = tl.load(input_ptr)
        mean = total / n_elements
        tl.store(output_ptr, mean)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes Mean Squared Error loss using Triton kernel.
    
    Parameters:
        predictions: Predicted tensor (same shape as targets)
        targets: Target tensor (same shape as predictions)
    
    Returns:
        Scalar MSE loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure predictions and targets have the same shape
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    n_elements = predictions.numel()
    
    # For the first kernel, we need a single atomic accumulator
    # Use a small buffer to hold the partial sums
    # We'll use a grid that allows one block to accumulate into a single location
    
    # Allocate output buffer for the sum
    # Since we're using atomic_add, we need to be careful about the grid design
    # A better approach is to use a two-phase reduction:
    # Phase 1: Each block computes a partial sum
    # Phase 2: Sum all partial sums and divide by n_elements
    
    # Phase 1: Compute partial sums per block
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For very small n_elements, we might want to use just one block
    # But to keep it simple, we'll use a reduction approach
    
    # Create buffer for partial sums - one element per block
    partial_sums = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
    
    # Launch the first kernel
    mse_loss_kernel[(num_blocks,)](
        predictions, 
        targets, 
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Sum the partial sums
    # Use another kernel to sum the partial sums
    final_BLOCK_SIZE = 256
    final_num_blocks = (num_blocks + final_BLOCK_SIZE - 1) // final_BLOCK_SIZE
    
    # For the final reduction, we can use a simple kernel
    if final_num_blocks > 1:
        # Multiple blocks needed for final reduction
        temp_buffer = torch.zeros(final_num_blocks, device=predictions.device, dtype=predictions.dtype)
        final_reduce_kernel[(final_num_blocks,)](
            partial_sums,
            temp_buffer,
            num_blocks,
            BLOCK_SIZE=final_BLOCK_SIZE
        )
        
        # Continue reducing until we get to one element
        while final_num_blocks > 1:
            partial_sums = temp_buffer
            num_blocks = final_num_blocks
            final_num_blocks = (num_blocks + final_BLOCK_SIZE - 1) // final_BLOCK_SIZE
            temp_buffer = torch.zeros(final_num_blocks, device=predictions.device, dtype=predictions.dtype)
            final_reduce_kernel[(final_num_blocks,)](
                partial_sums,
                temp_buffer,
                num_blocks,
                BLOCK_SIZE=final_BLOCK_SIZE
            )
        
        final_result = temp_buffer[0]
    else:
        # Only one block, so the result is just partial_sums[0]
        final_result = partial_sums[0]
    
    # Compute mean by dividing by n_elements
    result = final_result / n_elements
    
    return result


# For better performance, let's use a more efficient single-kernel approach
# that leverages Triton's reduction capabilities

@triton.jit
def mse_loss_fused_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Use reduction pattern for efficient sum computation
    # Initialize accumulator
    sum = tl.zeros([1], dtype=tl.float32)
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and compute squared differences
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Accumulate
    sum += tl.sum(squared_diff, axis=0)
    
    # Store partial sum
    tl.atomic_add(output_ptr, sum, mask=tl.arange(0, 1) == 0)


def triton_mse_loss_optimized(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Optimized MSE loss using Triton with fused reduction.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    n_elements = predictions.numel()
    
    # Use a simple approach: compute in float32 and use reduction
    BLOCK_SIZE = 256
    num_blocks = min(1024, (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Create output buffer with size 1
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    # Launch kernel
    mse_loss_fused_kernel[(num_blocks,)](
        predictions,
        targets,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # The output now contains the sum, divide by n_elements to get mean
    return output[0] / n_elements


# Even better: Use a direct reduction with Triton's built-in support
@triton.jit
def mse_loss_final_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel performs a complete reduction to compute MSE
    # Initialize accumulator
    sum = tl.zeros([1], dtype=tl.float32)
    
    # Loop over the input in strides
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared difference
        diff = predictions - targets
        squared_diff = diff * diff
        
        # Accumulate
        sum += tl.sum(squared_diff, axis=0)
    
    # Store the sum
    tl.store(output_ptr, sum)


def triton_mse_loss_final(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Final optimized MSE loss implementation using Triton.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    n_elements = predictions.numel()
    
    # Use a single block approach for simplicity and efficiency
    BLOCK_SIZE = 1024
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    # Launch kernel
    mse_loss_final_kernel[(1,)](
        predictions,
        targets,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean
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
        # Call our optimized Triton-based MSE loss
        return triton_mse_loss_final(predictions, targets)