import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr, out_ptr,
    B, D, margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (sample in the batch)
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    # Pointers to the start of the row for each tensor
    # Tensors are assumed to be contiguous
    a_row_ptr = anchor_ptr + row_idx * D
    p_row_ptr = positive_ptr + row_idx * D
    n_row_ptr = negative_ptr + row_idx * D

    dist_p_sq = 0.0
    dist_n_sq = 0.0

    # Loop over the feature dimension D in chunks of BLOCK_SIZE
    # to compute the squared L2 norm efficiently
    for i in range(0, D, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        
        # Load data for the current block
        a = tl.load(a_row_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared differences and accumulate
        diff_p = a - p
        diff_n = a - n
        dist_p_sq += tl.sum(diff_p * diff_p, axis=0)
        dist_n_sq += tl.sum(diff_n * diff_n, axis=0)

    # Calculate L2 norms (Euclidean distance)
    dist_p = tl.sqrt(dist_p_sq)
    dist_n = tl.sqrt(dist_n_sq)
    
    # Compute the triplet margin loss for this sample: max(0, dist_p - dist_n + margin)
    loss = tl.maximum(0.0, dist_p - dist_n + margin)
    
    # Store the per-sample loss in the output tensor
    tl.store(out_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    # Ensure inputs are contiguous and on GPU for Triton kernels
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    B, D = anchor.shape
    # The output tensor stores the loss for each sample in the batch
    out = torch.empty((B,), device=anchor.device, dtype=anchor.dtype)
    
    # BLOCK_SIZE is the number of elements processed per iteration in the loop.
    # 1024 is a balanced choice for feature dimensions like 8192.
    BLOCK_SIZE = 1024
    
    # Grid is one program per sample in the batch
    grid = (B,)
    
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out,
        B, D, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # PyTorch's TripletMarginLoss default reduction is 'mean'
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using a custom Triton kernel
    to fuse distance calculations and the margin operation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use the custom Triton implementation instead of torch.nn.TripletMarginLoss
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)