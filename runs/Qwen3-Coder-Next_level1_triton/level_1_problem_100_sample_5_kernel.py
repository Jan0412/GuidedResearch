import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output (mean loss)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulator for the sum
    sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute hinge loss: max(0, 1 - predictions * targets)
        loss = tl.maximum(0.0, 1.0 - predictions * targets)
        sum += tl.sum(loss, axis=0)
    
    # Compute mean and store result
    mean = sum / n_elements
    tl.store(output_ptr, mean)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Hinge Loss using Triton kernel.
    
    Args:
        predictions: Tensor of shape (batch_size,) with model predictions
        targets: Tensor of shape (batch_size,) with binary targets in {-1, 1}
    
    Returns:
        Scalar tensor with mean hinge loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Ensure targets match predictions shape
    if targets.shape != predictions.shape:
        targets = targets.view(-1)
        assert targets.shape == predictions.shape, "Targets must match predictions shape"
    
    # Create output tensor
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    
    # Configure kernel
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Launch kernel with single block for reduction
    # Since we're doing reduction, we use a single block and let Triton handle it
    # For very large tensors, we could use a two-stage reduction, but for simplicity
    # we use a single kernel with block-level reduction
    
    # Actually, for proper reduction, we need to handle it differently
    # Let's use a more efficient approach with multiple blocks and then reduce
    
    grid = lambda meta: (min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1024),)
    
    # For very large tensors, we might need multiple passes, but for now
    # we'll use a simple approach that works for most cases
    
    # Use a temporary buffer for partial sums if needed
    if n_elements > BLOCK_SIZE * 1024:
        # Use two-stage reduction
        temp_sum = torch.empty((triton.cdiv(n_elements, BLOCK_SIZE),), device=predictions.device, dtype=torch.float32)
        
        # First stage: compute partial sums
        _hinge_loss_stage1[temp_sum.shape](predictions, targets, temp_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        # Second stage: reduce partial sums
        _hinge_loss_stage2[(1,)](temp_sum, output, temp_sum.shape[0], BLOCK_SIZE=1024)
    else:
        # Single stage reduction
        _hinge_loss_stage1[(1,)](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


@triton.jit
def _hinge_loss_stage1(
    predictions_ptr,
    targets_ptr,
    output_ptr,  # Pointer to partial sums
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr = 1024,
):
    # Each block computes a partial sum
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = start + i
        if idx < n_elements:
            predictions = tl.load(predictions_ptr + idx)
            targets = tl.load(targets_ptr + idx)
            loss = tl.maximum(0.0, 1.0 - predictions * targets)
            sum += loss
    
    # Store partial sum
    tl.store(output_ptr + block_id, sum)


@triton.jit
def _hinge_loss_stage2(
    partial_sums_ptr,
    output_ptr,
    n_partial_sums,
    BLOCK_SIZE: tl.constexpr,
):
    # Final reduction of partial sums
    sum = tl.zeros((1,), dtype=tl.float32)
    
    for start in range(0, n_partial_sums, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_partial_sums
        partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        sum += tl.sum(partial_sums, axis=0)
    
    # Compute final mean
    # We need total elements from first stage, but for simplicity,
    # we'll just return the sum of partial sums divided by total elements
    # This is a limitation - we need to pass total elements
    # For now, we'll assume it's passed differently or use a workaround
    
    # Actually, let's use a simpler approach - just do single-block reduction
    tl.store(output_ptr, sum)


# Simplified version that works well for typical sizes
def triton_hinge_loss_simple(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Simplified Hinge Loss using Triton kernel.
    Works well for typical batch sizes up to millions.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Reshape targets to match predictions if needed
    targets = targets.view(-1)
    assert targets.shape == predictions.shape, "Targets must match predictions shape"
    
    # Create output tensor
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    
    # Use a reasonable block size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # For reduction, we need to use a different approach
    # Let's use the triton reduction pattern
    
    # Actually, let's implement a cleaner version
    return _hinge_loss_compute(predictions, targets, n_elements)


@triton.jit
def _hinge_loss_compute_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Reduction kernel for hinge loss
    # Each program handles a segment and accumulates into a sum
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Calculate segment boundaries
    segment_size = (n_elements + num_programs - 1) // num_programs
    start = pid * segment_size
    end = min(start + segment_size, n_elements)
    
    # Accumulate sum for this segment
    sum = tl.zeros((1,), dtype=tl.float32)
    for i in range(start, end):
        if i < n_elements:
            predictions = tl.load(predictions_ptr + i)
            targets = tl.load(targets_ptr + i)
            loss = tl.maximum(0.0, 1.0 - predictions * targets)
            sum += loss
    
    # Store partial sum
    tl.store(output_ptr + pid, sum)


@triton.jit
def _reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple reduction kernel
    sum = tl.zeros((1,), dtype=tl.float32)
    
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        sum += tl.sum(x, axis=0)
    
    tl.store(output_ptr, sum)


def _hinge_loss_compute(predictions: torch.Tensor, targets: torch.Tensor, n_elements: int):
    """Compute hinge loss with optimized Triton kernels."""
    # First pass: compute partial sums
    BLOCK_SIZE = 256
    num_blocks = min(triton.cdiv(n_elements, BLOCK_SIZE), 1024)
    
    partial_sums = torch.empty((num_blocks,), device=predictions.device, dtype=torch.float32)
    
    # Launch first kernel
    _hinge_loss_compute_kernel[(num_blocks,)](
        predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second pass: reduce partial sums
    final_sum = torch.empty((1,), device=predictions.device, dtype=torch.float32)
    _reduce_sum_kernel[(1,)](partial_sums, final_sum, num_blocks, BLOCK_SIZE=1024)
    
    # Compute mean
    mean_loss = final_sum / n_elements
    
    return mean_loss


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return _hinge_loss_compute(predictions, targets, predictions.numel())