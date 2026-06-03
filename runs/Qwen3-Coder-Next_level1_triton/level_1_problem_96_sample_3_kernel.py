import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    x_ptr,  # predictions
    y_ptr,  # targets
    n_elements,  # total number of elements
    output_ptr,  # output pointer (single element for mean)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = x - y
    abs_diff = tl.abs(diff)
    
    # Smooth L1 loss: 
    # if |diff| < 1: 0.5 * diff^2
    # else: |diff| - 0.5
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    # Accumulate using reduction
    # Use tl.sum for the block
    block_sum = tl.sum(loss, axis=0)
    
    # Accumulate to global sum using atomic add
    # For simplicity, we'll use a single block approach for the reduction
    # or we can do a two-pass approach. For this implementation, we'll 
    # use a simple reduction pattern that works well for large n_elements
    
    # Since we need to compute the mean, we'll accumulate all values
    # and then divide by n_elements at the end
    
    # For this implementation, we'll use a simple approach where
    # each block writes to a temporary array, then we do a second kernel
    # or use a simpler single-block approach for small to medium sizes.
    # But for generality, let's implement a proper reduction.
    
    # Actually, for simplicity and performance, let's use a single block 
    # reduction when n_elements <= BLOCK_SIZE * 1024, otherwise use two-pass
    # But since this is for a single scalar output, let's implement 
    # a simple atomic accumulation approach.
    
    # Write to global accumulator
    tl.atomic_add(output_ptr, block_sum, mask=mask)
    

@triton.jit
def smooth_l1_loss_kernel_final(
    x_ptr,  # predictions
    y_ptr,  # targets
    n_elements,  # total number of elements
    output_ptr,  # output pointer (single element for mean)
    BLOCK_SIZE: tl.constexpr,
):
    # Simple approach: one block computes the full reduction
    # This is efficient for reasonable sizes
    
    # Load data
    offset = tl.program_id(0) * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    diff = x - y
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    # Reduce within block
    for i in range(BLOCK_SIZE // 2):
        stride = BLOCK_SIZE // (2 ** i)
        if stride > 0:
            other_offsets = offsets + stride
            other_mask = other_offsets < n_elements
            other_loss = tl.load(loss + other_offsets, mask=other_mask, other=0.0)
            mask = mask & (other_offsets < n_elements)
            loss = tl.where(mask, loss + other_loss, loss)
    
    # Write result
    if tl.program_id(0) == 0:
        final_sum = tl.sum(loss, axis=0)
        tl.store(output_ptr, final_sum / n_elements)


# Actually, let's use a better approach with triton's reduction
@triton.jit
def smooth_l1_loss_reduce_kernel(
    x_ptr,  # predictions
    y_ptr,  # targets
    n_elements,  # total number of elements
    output_ptr,  # output pointer (single element for mean)
    BLOCK_SIZE: tl.constexpr,
):
    # Use tl.reduce for better performance
    sum = 0.0
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        
        diff = x - y
        abs_diff = tl.abs(diff)
        loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        
        # Accumulate
        sum += tl.sum(loss, axis=0)
    
    # Store mean
    tl.store(output_ptr, sum / n_elements)


# Let's implement a more efficient version with proper parallel reduction
@triton.jit
def smooth_l1_loss_kernel(
    x_ptr,  # predictions
    y_ptr,  # targets
    n_elements,  # total number of elements
    output_ptr,  # output pointer (single element for mean)
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each block computes a partial sum
    sum = tl.zeros([1], dtype=tl.float32)
    
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        
        diff = x - y
        abs_diff = tl.abs(diff)
        loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        
        sum += tl.sum(loss, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), sum)


# Better approach: fused kernel with atomic add for the final reduction
@triton.jit
def smooth_l1_loss_kernel_atomic(
    x_ptr,  # predictions
    y_ptr,  # targets
    n_elements,  # total number of elements
    output_ptr,  # output pointer (single element for mean)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = x - y
    abs_diff = tl.abs(diff)
    
    # Smooth L1 loss
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    # Accumulate block sum
    block_sum = tl.sum(loss, axis=0)
    
    # Use atomic add to accumulate to global sum
    tl.atomic_add(output_ptr, block_sum)


def triton_smooth_l1_loss(predictions, targets):
    """
    Triton implementation of Smooth L1 (Huber) Loss with mean reduction.
    
    Args:
        predictions: predicted values tensor
        targets: target values tensor
    Returns:
        scalar tensor with mean smooth l1 loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Input shapes must match."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    if n_elements == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # For the final mean, we'll use atomic accumulation
    # Create output tensor for accumulating the sum
    # Use a small buffer for atomic accumulation
    output_sum = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel to compute partial sums and accumulate
    smooth_l1_loss_kernel_atomic[grid](
        predictions, targets, n_elements, output_sum, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean
    return output_sum[0] / n_elements


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    Optimized with Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)