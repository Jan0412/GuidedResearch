import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr, output_ptr,
    batch_size, dim,
    p: tl.constexpr, margin: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes one batch element
    batch_id = tl.program_id(0)
    
    # Compute starting offsets for this batch element
    offset = batch_id * dim
    
    # Accumulator for distances
    anchor_pos_dist = 0.0
    anchor_neg_dist = 0.0
    
    # Process in blocks
    for start in range(0, dim, BLOCK_SIZE):
        block_offsets = offset + start + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < offset + dim
        
        # Load data
        a = tl.load(anchor_ptr + block_offsets, mask=mask, other=0.0)
        p_val = tl.load(positive_ptr + block_offsets, mask=mask, other=0.0)
        n_val = tl.load(negative_ptr + block_offsets, mask=mask, other=0.0)
        
        # Compute squared differences for L2 distance (p=2)
        if p == 2:
            diff_pos = a - p_val
            diff_neg = a - n_val
            anchor_pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
            anchor_neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
        else:
            # For general p, use absolute differences raised to power p
            diff_pos = tl.abs(a - p_val)
            diff_neg = tl.abs(a - n_val)
            anchor_pos_dist += tl.sum(tl.pow(diff_pos, p), axis=0)
            anchor_neg_dist += tl.sum(tl.pow(diff_neg, p), axis=0)
    
    # Compute final distances (root for p-norm)
    if p == 2:
        dist_pos = tl.sqrt(anchor_pos_dist)
        dist_neg = tl.sqrt(anchor_neg_dist)
    else:
        dist_pos = tl.pow(anchor_pos_dist, 1.0 / p)
        dist_neg = tl.pow(anchor_neg_dist, 1.0 / p)
    
    # Compute triplet loss: max(0, dist_pos - dist_neg + margin)
    loss = tl.maximum(0.0, dist_pos - dist_neg + margin)
    
    # Store result
    tl.store(output_ptr + batch_id, loss)


def triton_triplet_margin_loss(anchor, positive, negative, p=2.0, margin=1.0):
    """
    Compute triplet margin loss using Triton kernel.
    
    Parameters:
        anchor: anchor embeddings [batch_size, dim]
        positive: positive embeddings [batch_size, dim]
        negative: negative embeddings [batch_size, dim]
        p: p-norm to use (default: 2 for Euclidean)
        margin: margin for the loss
    
    Returns:
        Scalar tensor with the mean triplet loss
    """
    assert anchor.shape == positive.shape == negative.shape, "All inputs must have the same shape"
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    batch_size, dim = anchor.shape
    output = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Determine block size based on dimension
    BLOCK_SIZE = 256
    
    # Launch kernel for each batch element
    grid = (batch_size,)
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, output,
        batch_size, dim,
        p=int(p), margin=margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss across batch
    return output.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, p=2.0, margin=self.margin)