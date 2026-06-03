import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr, 
    targets_ptr, 
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = predictions - targets
    
    # Compute smooth L1 loss: 
    # for |x| < 1: 0.5 * x^2
    # for |x| >= 1: |x| - 0.5
    abs_diff = tl.abs(diff)
    loss = tl.where(
        abs_diff < 1.0,
        0.5 * diff * diff,
        abs_diff - 0.5
    )
    
    # Store loss values
    tl.store(output_ptr + offsets, loss, mask=mask)


@triton.jit
def sum_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each block computes a partial sum
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and sum
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + pid, sum_val)


class SmoothL1LossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        # Ensure tensors are contiguous
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Get number of elements
        n_elements = predictions.numel()
        
        # Allocate intermediate loss tensor
        loss_buffer = torch.empty_like(predictions)
        
        # Compute smooth L1 loss per element
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        smooth_l1_loss_kernel[grid](
            predictions, targets, loss_buffer, 
            n_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # For the mean reduction, we'll use a simple approach:
        # Compute sum using Triton, then divide by n_elements
        # Since we need mean, we can do this efficiently
        
        # Compute sum of losses
        sum_result = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
        
        # Use a single block to sum all elements for small tensors or
        # use multiple blocks for larger ones
        if n_elements <= BLOCK_SIZE:
            # Single block reduction
            sum_reduce_kernel[(1,)](
                loss_buffer, sum_result,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # Two-step reduction: first reduce to chunks, then sum chunks
            num_chunks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            temp_sum = torch.empty(num_chunks, device=predictions.device, dtype=predictions.dtype)
            
            grid1 = lambda meta: ((num_chunks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
            # Actually, we want one block per chunk, so:
            grid1 = (num_chunks,)
            sum_reduce_kernel[grid1](
                loss_buffer, temp_sum,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Second reduction step on the partial sums
            sum_reduce_kernel[(1,)](
                temp_sum, sum_result,
                num_chunks,
                BLOCK_SIZE=BLOCK_SIZE
            )
        
        # Compute mean
        loss_mean = sum_result[0] / n_elements
        
        # Save for backward pass
        ctx.save_for_backward(predictions, targets)
        ctx.n_elements = n_elements
        
        return loss_mean
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        n_elements = ctx.n_elements
        
        # Gradient of smooth L1 loss w.r.t. predictions:
        # for |x| < 1: x (where x = predictions - targets)
        # for |x| >= 1: sign(x)
        # and then multiply by grad_output / n_elements (since it's mean)
        
        # Compute difference
        diff = predictions - targets
        abs_diff = tl.abs(diff)
        
        # Create gradient tensor
        grad_input = torch.empty_like(predictions)
        
        # Compute gradient for smooth L1
        # We'll use a simple kernel for this
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        
        # Define gradient kernel
        @triton.jit
        def smooth_l1_grad_kernel(
            diff_ptr, grad_output_ptr, 
            grad_input_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            # Load difference
            diff = tl.load(diff_ptr + offsets, mask=mask, other=0.0)
            
            # Compute gradient of smooth L1
            # For |x| < 1: x
            # For |x| >= 1: sign(x)
            grad = tl.where(
                tl.abs(diff) < 1.0,
                diff,
                tl.where(diff > 0.0, 1.0, -1.0)
            )
            
            # Multiply by grad_output / n_elements
            grad_input = tl.load(grad_output_ptr) * grad / n_elements
            
            # Store result
            tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
        
        smooth_l1_grad_kernel[grid](
            diff, grad_output,
            grad_input,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Gradient w.r.t. targets is the negative
        grad_target = -grad_input
        
        return grad_input, grad_target


def smooth_l1_loss_triton(predictions, targets):
    return SmoothL1LossFunc.apply(predictions, targets)


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return smooth_l1_loss_triton(predictions, targets)