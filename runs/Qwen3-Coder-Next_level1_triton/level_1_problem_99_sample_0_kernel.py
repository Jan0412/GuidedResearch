import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    # Process one sample per program for simplicity and memory coalescing
    batch_idx = tl.program_id(0)
    
    # Compute offsets for this batch element
    anchor_offset = batch_idx * dim
    positive_offset = batch_idx * dim
    negative_offset = batch_idx * dim
    output_offset = batch_idx
    
    # Accumulate squared differences for anchor-positive
    ap_sum = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, dim, BLOCK_SIZE):
        block_size_actual = tl.minimum(BLOCK_SIZE, dim - i)
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor and positive values
        a = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        
        diff = a - p
        ap_sum += diff * diff
    
    # Compute L2 distance (sqrt of sum of squares)
    ap_distance = tl.sqrt(ap_sum)
    
    # Accumulate squared differences for anchor-negative
    an_sum = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, dim, BLOCK_SIZE):
        block_size_actual = tl.minimum(BLOCK_SIZE, dim - i)
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor and negative values
        a = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        diff = a - n
        an_sum += diff * diff
    
    # Compute L2 distance (sqrt of sum of squares)
    an_distance = tl.sqrt(an_sum)
    
    # Compute triplet loss: max(ap_distance - an_distance + margin, 0)
    loss = tl.maximum(ap_distance - an_distance + margin, 0.0)
    
    # Store the result
    tl.store(output_ptr + output_offset, loss)


def triplet_margin_loss_triton(anchor, positive, negative, margin=1.0, block_size=128):
    """
    Compute triplet margin loss using Triton kernel.
    
    Args:
        anchor: [batch_size, dim]
        positive: [batch_size, dim]
        negative: [batch_size, dim]
        margin: margin value
        block_size: block size for Triton kernel
    
    Returns:
        Scalar tensor with the mean loss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    
    # Create output tensor for batch losses
    batch_losses = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, batch_losses,
        batch_size, dim, margin,
        BLOCK_SIZE=block_size
    )
    
    # Return mean loss
    return batch_losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for TripletMarginLoss computation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_margin_loss_triton(anchor, positive, negative, margin=self.margin)