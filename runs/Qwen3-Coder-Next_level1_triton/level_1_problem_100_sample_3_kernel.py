import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar (mean hinge loss)
    n_elements,       # Total number of elements (batch size)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - predictions * targets)
    margin = 1.0 - predictions * targets
    loss = tl.where(margin > 0.0, margin, 0.0)
    
    # Accumulate sum for mean calculation using atomic add to a global sum
    # Since we can't use atomic operations directly for float in all cases,
    # we'll use a reduction approach with a temporary buffer
    tl.store(loss_buffer_ptr + offsets, loss, mask=mask)


@triton.jit
def reduce_sum_kernel(
    input_ptr,        # Pointer to input tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel reduces a tensor to a single sum value
    # We'll use a two-pass approach: first reduce to blocks, then reduce blocks
    
    # For simplicity, we'll implement a simple parallel reduction
    # Each block computes a partial sum
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Reduce within the block
    for i in range(BLOCK_SIZE // 2):
        stride = BLOCK_SIZE // (2 * (i + 1))
        if stride > 0:
            offset1 = offsets
            offset2 = offsets + stride
            mask1 = offset1 < n_elements
            mask2 = offset2 < n_elements
            x = tl.where((offset1 < n_elements) & (offset2 < n_elements),
                        x + tl.load(input_ptr + offset2, mask=mask2, other=0.0),
                        x)
    
    # Store the result for this block
    tl.store(output_ptr + tl.program_id(0), x[0], mask=tl.program_id(0) < n_elements // BLOCK_SIZE + (1 if n_elements % BLOCK_SIZE > 0 else 0))


@triton.jit
def final_reduce_kernel(
    input_ptr,        # Pointer to partial sums
    output_ptr,       # Pointer to final output
    n_partial_sums,   # Number of partial sums
    BLOCK_SIZE: tl.constexpr,
):
    # Final reduction to get the total sum
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x)
    
    tl.store(output_ptr, sum_val)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute mean hinge loss using Triton kernels.
    
    Hinge loss = mean(max(0, 1 - predictions * targets))
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape[0] == targets.shape[0], "Batch size must match."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.numel()
    if batch_size == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # First, compute hinge loss for each element and store in temporary buffer
    temp_loss = torch.empty_like(predictions)
    
    # Define kernel with temp_loss_ptr
    @triton.jit
    def hinge_loss_kernel_with_temp(
        predictions_ptr,  # Pointer to predictions tensor
        targets_ptr,      # Pointer to targets tensor
        loss_ptr,         # Pointer to temporary loss buffer
        n_elements,       # Total number of elements (batch size)
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute hinge loss: max(0, 1 - predictions * targets)
        margin = 1.0 - predictions * targets
        loss = tl.where(margin > 0.0, margin, 0.0)
        
        # Store the result
        tl.store(loss_ptr + offsets, loss, mask=mask)
    
    # Launch the kernel
    hinge_loss_kernel_with_temp[grid](predictions, targets, temp_loss, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Now reduce the loss tensor to get the mean
    # We'll use a simple reduction approach with multiple blocks
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create a buffer for partial sums
    partial_sums = torch.empty(num_blocks, device=predictions.device)
    
    # Launch reduction kernel
    @triton.jit
    def block_reduce_kernel(
        input_ptr,        # Pointer to input tensor
        output_ptr,       # Pointer to output partial sums
        n_elements,       # Total number of elements
        BLOCK_SIZE: tl.constexpr,
        NUM_ELEMENTS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * NUM_ELEMENTS_PER_BLOCK
        
        # Accumulate sum for this block
        sum_val = 0.0
        for i in range(NUM_ELEMENTS_PER_BLOCK):
            offset = block_start + i
            if offset < n_elements:
                sum_val += tl.load(input_ptr + offset)
        
        # Store result for this block
        tl.store(output_ptr + pid, sum_val)
    
    # Launch block reduction
    block_reduce_kernel[(num_blocks,)](temp_loss, partial_sums, batch_size, BLOCK_SIZE=1, NUM_ELEMENTS_PER_BLOCK=BLOCK_SIZE)
    
    # Final reduction of partial sums
    final_sum = torch.empty(1, device=predictions.device)
    
    # Simple reduction for the final sums
    if num_blocks <= 1024:
        final_block_size = 1024
    else:
        final_block_size = num_blocks
        
    @triton.jit
    def simple_reduce_kernel(
        input_ptr,        # Pointer to input tensor
        output_ptr,       # Pointer to output scalar
        n_elements,       # Total number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        sum_val = tl.sum(x)
        
        tl.store(output_ptr, sum_val)
    
    # Launch final reduction
    simple_reduce_kernel[(1,)](partial_sums, final_sum, num_blocks, BLOCK_SIZE=final_block_size)
    
    # Compute mean
    mean_loss = final_sum[0] / batch_size
    
    return mean_loss


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our custom Triton implementation
        return triton_hinge_loss(predictions, targets)