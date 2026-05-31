import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_loss_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    B, D, margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one sample in the batch)
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    # Pointers for the current row in the batch
    a_ptr = anchor_ptr + row_idx * D
    p_ptr = pos_ptr + row_idx * D
    n_ptr = neg_ptr + row_idx * D

    sum_sq_pos = 0.0
    sum_sq_neg = 0.0

    # Iterate over the feature dimension in blocks
    for i in range(0, D, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        
        # Load feature vectors
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + offsets, mask=mask, other=0.0)
        
        # Calculate squared Euclidean distance components
        diff_p = a - p
        diff_n = a - n
        sum_sq_pos += tl.sum(diff_p * diff_p, axis=0)
        sum_sq_neg += tl.sum(diff_n * diff_n, axis=0)

    # Compute the Euclidean distances
    dist_pos = tl.sqrt(sum_sq_pos)
    dist_neg = tl.sqrt(sum_sq_neg)
    
    # Triplet Margin Loss: max(d(a, p) - d(a, n) + margin, 0)
    loss = tl.maximum(dist_pos - dist_neg + margin, 0.0)
    
    # Store the per-sample loss
    tl.store(out_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Triton-optimized wrapper for Triplet Margin Loss.
    """
    # Ensure inputs are contiguous on GPU for the kernel
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    B, D = anchor.shape
    # Output buffer to store the loss for each sample in the batch
    out = torch.empty((B,), device=anchor.device, dtype=anchor.dtype)
    
    # BLOCK_SIZE is the number of elements processed per iteration in the feature dimension
    BLOCK_SIZE = 1024
    grid = (B,)
    
    # Launch the Triton kernel
    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        B, D, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean loss across the batch to match PyTorch's default reduction='mean'
    return torch.mean(out)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace the standard TripletMarginLoss with the Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)