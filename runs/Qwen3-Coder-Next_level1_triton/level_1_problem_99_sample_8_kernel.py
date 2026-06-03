import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    batch_size,
    dim,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Compute the starting offsets for this batch element
    anchor_offset = batch_idx * dim
    positive_offset = batch_idx * dim
    negative_offset = batch_idx * dim
    
    # Accumulators for squared distances
    pos_dist = 0.0
    neg_dist = 0.0
    
    # Iterate over dimensions in blocks
    for start_d in range(0, dim, BLOCK_SIZE):
        d_offsets = start_d + tl.arange(0, BLOCK_SIZE)
        mask = d_offsets < dim
        
        # Load values
        a = tl.load(anchor_ptr + anchor_offset + d_offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + d_offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + d_offsets, mask=mask, other=0.0)
        
        # Compute squared differences for positive and negative pairs
        diff_pos = a - p
        diff_neg = a - n
        pos_dist += tl.sum(diff_pos * diff_pos)
        neg_dist += tl.sum(diff_neg * diff_neg)
    
    # Compute margin loss for this sample
    loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)


@triton.jit
def triplet_margin_loss_backward_kernel(
    grad_output_ptr,
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    grad_anchor_ptr,
    grad_positive_ptr,
    grad_negative_ptr,
    batch_size,
    dim,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Compute the starting offsets for this batch element
    anchor_offset = batch_idx * dim
    positive_offset = batch_idx * dim
    negative_offset = batch_idx * dim
    
    # Accumulators for gradients
    grad_a_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_p_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_n_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Check if this sample contributed to the loss (pos_dist - neg_dist + margin > 0)
    pos_dist = 0.0
    neg_dist = 0.0
    
    # First pass: compute distances
    for start_d in range(0, dim, BLOCK_SIZE):
        d_offsets = start_d + tl.arange(0, BLOCK_SIZE)
        mask = d_offsets < dim
        
        a = tl.load(anchor_ptr + anchor_offset + d_offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + d_offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + d_offsets, mask=mask, other=0.0)
        
        diff_pos = a - p
        diff_neg = a - n
        pos_dist += tl.sum(diff_pos * diff_pos)
        neg_dist += tl.sum(diff_neg * diff_neg)
    
    # Check if margin loss is active
    is_active = (pos_dist - neg_dist + margin) > 0
    grad_scale = tl.where(is_active, tl.load(grad_output_ptr), 0.0)
    
    # Second pass: compute gradients
    for start_d in range(0, dim, BLOCK_SIZE):
        d_offsets = start_d + tl.arange(0, BLOCK_SIZE)
        mask = d_offsets < dim
        
        a = tl.load(anchor_ptr + anchor_offset + d_offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + d_offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + d_offsets, mask=mask, other=0.0)
        
        # Compute gradients for active samples
        # d(pos_dist)/d(anchor) = 2*(anchor - positive)
        # d(pos_dist)/d(positive) = -2*(anchor - positive)
        # d(neg_dist)/d(anchor) = 2*(anchor - negative)
        # d(neg_dist)/d(negative) = -2*(anchor - negative)
        
        diff_pos = a - p
        diff_neg = a - n
        
        grad_a = 2.0 * (diff_pos - diff_neg)
        grad_p = -2.0 * diff_pos
        grad_n = -2.0 * diff_neg
        
        grad_a_sum += grad_a * grad_scale
        grad_p_sum += grad_p * grad_scale
        grad_n_sum += grad_n * grad_scale
    
    # Store gradients
    for start_d in range(0, dim, BLOCK_SIZE):
        d_offsets = start_d + tl.arange(0, BLOCK_SIZE)
        mask = d_offsets < dim
        
        tl.store(grad_anchor_ptr + anchor_offset + d_offsets, tl.load(grad_anchor_ptr + anchor_offset + d_offsets) + tl.sum(grad_a_sum, axis=0) * mask, mask=mask)
        tl.store(grad_positive_ptr + positive_offset + d_offsets, tl.load(grad_positive_ptr + positive_offset + d_offsets) + tl.sum(grad_p_sum, axis=0) * mask, mask=mask)
        tl.store(grad_negative_ptr + negative_offset + d_offsets, tl.load(grad_negative_ptr + negative_offset + d_offsets) + tl.sum(grad_n_sum, axis=0) * mask, mask=mask)


def triplet_margin_loss_triton(anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of TripletMarginLoss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1] if len(anchor.shape) > 1 else 1
    
    # For the forward pass, we compute loss per sample
    loss_per_sample = torch.empty(batch_size, dtype=anchor.dtype, device=anchor.device)
    
    BLOCK_SIZE = 256
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, loss_per_sample,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return torch.mean(loss_per_sample)


def triplet_margin_loss_backward_triton(grad_output, anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of TripletMarginLoss backward pass
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1] if len(anchor.shape) > 1 else 1
    
    # Initialize gradient tensors
    grad_anchor = torch.zeros_like(anchor)
    grad_positive = torch.zeros_like(positive)
    grad_negative = torch.zeros_like(negative)
    
    BLOCK_SIZE = 256
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_backward_kernel[grid](
        grad_output, anchor, positive, negative,
        grad_anchor, grad_positive, grad_negative,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return grad_anchor, grad_positive, grad_negative


class TripletMarginLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor, positive, negative, margin):
        # Save for backward pass
        ctx.save_for_backward(anchor, positive, negative)
        ctx.margin = margin
        
        # Forward pass
        loss = triplet_margin_loss_triton(anchor, positive, negative, margin)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        anchor, positive, negative = ctx.saved_tensors
        margin = ctx.margin
        
        # Compute gradients using our custom kernel
        grad_anchor, grad_positive, grad_negative = triplet_margin_loss_backward_triton(
            grad_output, anchor, positive, negative, margin
        )
        
        return grad_anchor, grad_positive, grad_negative, None


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for TripletMarginLoss.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use our custom autograd function
        return TripletMarginLossFunction.apply(anchor, positive, negative, self.margin)