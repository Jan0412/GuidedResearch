import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output (scalar result)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes Hinge Loss: mean(max(0, 1 - predictions * targets))
    This kernel fuses all operations into a single pass for efficiency.
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute hinge loss components: max(0, 1 - predictions * targets)
    loss = tl.maximum(0.0, 1.0 - predictions * targets)

    # Compute partial sum for this block
    block_sum = tl.sum(loss)
    block_count = tl.sum(mask.to(tl.float32))

    # Store partial results for final reduction
    # This kernel is designed to be called with a second reduction step
    # For simplicity, we'll use atomic operations for accumulation
    # But since we need a final mean, we'll do a two-pass approach:
    # First pass: compute sum and count
    # Second pass: compute mean (handled in Python wrapper)

    # For this implementation, we'll compute sum and count in the kernel
    # and store them for the final reduction in the Python wrapper
    # However, for a single kernel approach, we can use the following pattern:
    
    # Since Triton doesn't support global reduction directly in all versions,
    # we'll implement a simple approach where we compute the sum and count
    # and then do the final reduction in Python
    
    # But to keep it single kernel, we can use a different approach:
    # Use a temporary buffer for partial sums, but that's complex
    
    # Simpler approach: compute sum and count, store in output
    # Since this is a single output, we'll use atomic_add to accumulate
    # However, this is inefficient for large blocks
    
    # Better approach: use a two-kernel solution or just compute in Python
    # Given the constraints, let's use the standard pattern:
    # This kernel computes partial sums, then Python does final reduction
    
    # For simplicity and performance, we'll compute the sum here and return it
    # But since we need a single output value, we'll use a different strategy:
    # Use tl.atomic_add to accumulate into a single location
    # However, tl.atomic_add is available for float in newer Triton versions
    
    # Let's use a simpler approach: compute the sum and count, then Python does final reduction
    # But the problem requires a single output, so we need a reduction
    
    # For this specific case, let's implement the kernel to compute sum and count
    # and then use a separate reduction kernel or Python for final mean
    
    # Since the problem asks for a single optimized model, let's use a practical approach:
    # The kernel will compute the sum and count, and we'll handle the mean in Python
    pass


# Actually, let's implement a proper kernel with atomic operations for reduction
@triton.jit
def hinge_loss_kernel_atomic(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output (scalar result)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes Hinge Loss using atomic operations for reduction.
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute hinge loss: max(0, 1 - predictions * targets)
    loss = tl.maximum(0.0, 1.0 - predictions * targets)

    # Compute partial sum for this block
    block_sum = tl.sum(loss)

    # Use atomic add to accumulate the sum
    # Since atomic_add for float is available in newer Triton versions
    tl.atomic_add(output_ptr, block_sum)


# For a more efficient implementation, let's use a two-pass approach
# But since the problem wants a single optimized solution, let's use a fused approach
# that computes sum and count, then Python does final division

# Actually, let's implement a cleaner version with just one kernel
@triton.jit
def hinge_loss_fused_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute partial sum
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Hinge loss: max(0, 1 - pred * target)
    loss = tl.maximum(0.0, 1.0 - predictions * targets)
    
    # Accumulate sum using atomic operations
    block_sum = tl.sum(loss)
    tl.atomic_add(output_ptr, block_sum)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes Hinge Loss using Triton kernel.
    
    Hinge Loss = mean(max(0, 1 - predictions * targets))
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure targets are the same shape as predictions (or broadcastable)
    # For the given architecture, targets should be broadcast to predictions shape
    if targets.shape != predictions.shape:
        targets = targets.view(-1).expand(predictions.numel()).reshape(predictions.shape)
    
    n_elements = predictions.numel()
    
    # Create output tensor for sum (initialized to zero)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    hinge_loss_fused_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean
    return output / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using fused Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)