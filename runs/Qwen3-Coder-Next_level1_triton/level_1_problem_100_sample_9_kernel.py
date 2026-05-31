import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
    OUT_PTR,          # Pointer to output (for partial sums in first pass)
    IS_FIRST_PASS: tl.constexpr,  # Whether this is the first pass (compute partial sums)
    NUM_PARTIALS: tl.constexpr,   # Number of partial sums (for second pass)
):
    # Compute global index for this thread
    pid = tl.program_id(0)
    
    # Compute block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - predictions * targets)
    loss = 1.0 - predictions * targets
    loss = tl.where(loss > 0, loss, 0.0)
    
    # If first pass, compute partial sums; if second pass, compute final result
    if IS_FIRST_PASS:
        # Compute partial sum for this block
        partial_sum = tl.sum(loss, axis=0)
        # Store to output
        tl.store(OUT_PTR + pid, partial_sum)
    else:
        # For second pass, compute final result
        # We need to compute the mean: sum(loss) / n_elements
        total_sum = tl.sum(loss, axis=0)
        # Since this is the second pass, we'll accumulate the partial sum
        # and divide by n_elements at the end in the host code
        # But for simplicity, we'll just store the sum here and handle division in host
        tl.store(OUT_PTR + pid, total_sum)


@triton.jit
def finalize_mean_kernel(
    partial_sums_ptr,  # Pointer to partial sums
    final_ptr,         # Pointer to final result
    n_partials,        # Number of partial sums
    n_elements,        # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    
    # Load partial sums
    partials = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Compute total sum
    total_sum = tl.sum(partials, axis=0)
    
    # Compute mean
    mean = total_sum / n_elements
    
    # Store result
    tl.store(final_ptr, mean)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute hinge loss using Triton kernels.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Ensure targets is broadcastable to predictions shape
    # targets is (batch_size,), predictions is (batch_size, *input_shape)
    # For our case, input_shape is (32768,), so predictions is (32768, 32768)
    # But looking at get_inputs, it's (32768,) and (32768,) with batch_size=32768
    # So predictions and targets should be 1D with same size
    
    # Reshape targets to match predictions shape if needed
    if targets.shape != predictions.shape:
        targets = targets.view(-1)
        if targets.numel() != n_elements:
            # Expand targets to match predictions if needed
            targets = targets.expand_as(predictions).contiguous()
    
    # First, compute partial sums for the reduction
    # Use a reasonable block size
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate buffer for partial sums
    partial_sums = torch.empty(num_blocks, device=predictions.device, dtype=torch.float32)
    
    # First pass: compute partial sums
    grid = lambda meta: (num_blocks,)
    hinge_loss_kernel[grid](
        predictions, targets, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_PTR=partial_sums,
        IS_FIRST_PASS=True,
        NUM_PARTIALS=num_blocks
    )
    
    # Second pass: finalize the mean
    # We can do this with another kernel or just on CPU for small num_blocks
    # For simplicity and correctness, use torch.sum on partial_sums
    total_sum = torch.sum(partial_sums)
    result = total_sum / n_elements
    
    return result


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Ensure predictions and targets are float32 on CUDA
        predictions = predictions.float().contiguous()
        targets = targets.float().contiguous()
        
        # Call the optimized Triton hinge loss
        return triton_hinge_loss(predictions, targets)