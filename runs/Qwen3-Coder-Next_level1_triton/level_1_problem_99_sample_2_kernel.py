import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    n_elements, dim,
    margin, p_norm,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)
    
    # Calculate offsets for this sample
    offset_start = batch_idx * dim
    
    # Initialize accumulators for squared differences
    pos_dist_sq = 0.0
    neg_dist_sq = 0.0
    
    # Process in chunks to handle large dimensions
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block in range(num_blocks):
        block_offset = block * BLOCK_SIZE
        offsets = offset_start + block_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (batch_idx + 1) * dim
        
        # Load anchor, positive, negative values
        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        pos_diff = a - p
        neg_diff = a - n
        
        pos_dist_sq += tl.sum(pos_diff * pos_diff, mask=mask)
        neg_dist_sq += tl.sum(neg_diff * neg_diff, mask=mask)
    
    # Compute distances (Euclidean)
    pos_dist = tl.sqrt(pos_dist_sq)
    neg_dist = tl.sqrt(neg_dist_sq)
    
    # Compute triplet loss: max(0, pos_dist - neg_dist + margin)
    loss = tl.maximum(0.0, pos_dist - neg_dist + margin)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0):
    """
    Compute triplet margin loss using Triton kernel.
    
    Parameters:
        anchor, positive, negative: Input tensors of shape (batch_size, dim)
        margin: The margin value
        p: The norm degree (only p=2 is supported in this implementation)
    
    Returns:
        Scalar tensor with the average loss
    """
    assert anchor.shape == positive.shape == negative.shape, "All inputs must have the same shape"
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    
    # Output tensor for per-sample losses
    output = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Configure kernel
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative,
        output,
        anchor.numel(), dim,
        margin, p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return output.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use custom Triton kernel for triplet margin loss
        return triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin)