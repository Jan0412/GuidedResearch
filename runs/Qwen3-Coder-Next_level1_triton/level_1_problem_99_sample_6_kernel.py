import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    batch_size, dim,
    margin,
    BLOCK_SIZE: tl.constexpr
):
    # Compute L2 distance squared for each sample in batch
    batch_id = tl.program_id(0)
    
    # Pointers to current sample's data
    anchor_offset = batch_id * dim
    positive_offset = batch_id * dim
    negative_offset = batch_id * dim
    
    # Accumulate squared differences
    dist_pos_sq = 0.0
    dist_neg_sq = 0.0
    
    # Process in blocks to handle arbitrary dimensions
    for start_dim in range(0, dim, BLOCK_SIZE):
        # Create offsets for this block
        offsets = start_dim + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor, positive, negative values
        anchor_val = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_val = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        negative_val = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_pos = anchor_val - positive_val
        diff_neg = anchor_val - negative_val
        
        dist_pos_sq += tl.sum(diff_pos * diff_pos, axis=0)
        dist_neg_sq += tl.sum(diff_neg * diff_neg, axis=0)
    
    # Compute distances (sqrt)
    dist_pos = tl.sqrt(dist_pos_sq)
    dist_neg = tl.sqrt(dist_neg_sq)
    
    # Compute triplet margin loss
    loss = tl.maximum(0.0, dist_pos - dist_neg + margin)
    
    # Store result
    tl.store(output_ptr + batch_id, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of TripletMarginLoss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "Input tensors must have the same shape."
    
    batch_size, dim = anchor.shape[0], anchor.shape[1] if len(anchor.shape) > 1 else 1
    
    # Ensure contiguous memory
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    # Output tensor for per-sample losses
    output = torch.empty(batch_size, dtype=anchor.dtype, device=anchor.device)
    
    # Determine block size - tuned for good performance
    BLOCK_SIZE = 256
    
    # Launch kernel with one block per sample in batch
    grid = (batch_size,)
    
    # Launch the Triton kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative,
        output,
        batch_size, dim,
        margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss across batch (matching PyTorch's default behavior)
    return output.mean()


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for TripletMarginLoss.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)