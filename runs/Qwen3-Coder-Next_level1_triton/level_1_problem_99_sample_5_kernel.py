import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    n_elements, batch_size,
    margin,
    BLOCK_SIZE: tl.constexpr
):
    # Compute per-sample loss, so we process batch_size elements
    # Each block processes multiple samples to ensure good occupancy
    
    # We'll process one sample per program for simplicity and efficiency
    sample_idx = tl.program_id(0)
    
    if sample_idx >= batch_size:
        return
    
    # Compute offsets for this sample
    base_offset = sample_idx * n_elements
    
    # Compute squared differences for positive pair
    sum_pos_sq = 0.0
    sum_neg_sq = 0.0
    
    # Process in blocks to handle large dimensions
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = base_offset + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (base_offset + n_elements)
        
        # Load anchor, positive, negative values for this block
        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_pos = a - p
        diff_neg = a - n
        sum_pos_sq += tl.sum(diff_pos * diff_pos)
        sum_neg_sq += tl.sum(diff_neg * diff_neg)
    
    # Compute Euclidean distances (sqrt of sum of squares)
    pos_dist = tl.sqrt(sum_pos_sq)
    neg_dist = tl.sqrt(sum_neg_sq)
    
    # Compute triplet loss: max(0, pos_dist - neg_dist + margin)
    loss = tl.maximum(0.0, pos_dist - neg_dist + margin)
    
    # Store result (average over batch)
    tl.store(output_ptr + sample_idx, loss)


def triplet_margin_loss_triton(anchor, positive, negative, margin=1.0):
    """
    Compute triplet margin loss using Triton kernel.
    
    Args:
        anchor: anchor embeddings of shape (batch_size, dim)
        positive: positive embeddings of shape (batch_size, dim)
        negative: negative embeddings of shape (batch_size, dim)
        margin: margin value for triplet loss
    
    Returns:
        scalar tensor with the mean triplet loss across the batch
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    n_elements = dim
    
    # Output tensor for per-sample losses
    per_sample_loss = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Determine block size - use a reasonable size for the dimension
    BLOCK_SIZE = 256
    
    # Grid: one block per sample
    grid = (batch_size,)
    
    # Launch the kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative,
        per_sample_loss,
        n_elements, batch_size,
        margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss across batch
    return per_sample_loss.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_margin_loss_triton(anchor, positive, negative, self.margin)