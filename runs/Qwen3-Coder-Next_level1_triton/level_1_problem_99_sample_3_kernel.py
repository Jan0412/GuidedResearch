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
    eps,
    BLOCK_SIZE: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    # Calculate offsets for this batch element
    base_offset = batch_idx * dim
    
    # Accumulate squared differences for anchor-positive and anchor-negative
    pos_dist_sq = 0.0
    neg_dist_sq = 0.0
    
    # Iterate over dimensions in blocks
    for dim_start in range(0, dim, DIM_BLOCK):
        dim_end = tl.minimum(dim_start + DIM_BLOCK, dim)
        dim_range = dim_start + tl.arange(0, DIM_BLOCK)
        mask = dim_range < dim
        
        # Load anchor, positive, negative values for this dimension range
        a = tl.load(anchor_ptr + base_offset + dim_range, mask=mask, other=0.0)
        p = tl.load(positive_ptr + base_offset + dim_range, mask=mask, other=0.0)
        n = tl.load(negative_ptr + base_offset + dim_range, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_pos = a - p
        diff_neg = a - n
        
        pos_dist_sq += tl.sum(diff_pos * diff_pos, axis=0)
        neg_dist_sq += tl.sum(diff_neg * diff_neg, axis=0)
    
    # Compute final distances (with epsilon for numerical stability)
    pos_dist = tl.sqrt(pos_dist_sq + eps)
    neg_dist = tl.sqrt(neg_dist_sq + eps)
    
    # Compute triplet margin loss: max(0, pos_dist - neg_dist + margin)
    loss = tl.maximum(0.0, pos_dist - neg_dist + margin)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, reduction='mean'):
    """
    Compute triplet margin loss using Triton kernel.
    
    Parameters:
        anchor, positive, negative: Input tensors of shape (batch_size, dim)
        margin: The margin value
        p: Norm degree (currently only p=2 is supported)
        eps: Small value for numerical stability
        reduction: 'mean' or 'none'
    """
    assert p == 2, "Only L2 distance is supported in this implementation"
    
    # Ensure contiguous tensors
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1]
    
    # Prepare output tensor
    output = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Calculate optimal block sizes
    # DIM_BLOCK should be a power of 2 and <= 1024
    DIM_BLOCK = min(128, triton.next_power_of_2(dim))
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, output,
        batch_size, dim, margin, eps,
        BLOCK_SIZE=1,  # Not used in this kernel design
        DIM_BLOCK=DIM_BLOCK,
    )
    
    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:  # 'none'
        return output


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using Triton kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin)