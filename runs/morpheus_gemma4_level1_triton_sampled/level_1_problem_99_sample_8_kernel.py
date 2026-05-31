import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    batch_size, dim, margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    row_id = tl.program_id(0)
    if row_id >= batch_size:
        return

    # Accumulators for squared Euclidean distances
    dist_ap_sq = 0.0
    dist_an_sq = 0.0
    
    # Loop over the feature dimension in blocks
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load values for the current block
        a = tl.load(anchor_ptr + row_id * dim + offsets, mask=mask, other=0.0)
        p = tl.load(pos_ptr + row_id * dim + offsets, mask=mask, other=0.0)
        n = tl.load(neg_ptr + row_id * dim + offsets, mask=mask, other=0.0)
        
        # Compute squared differences and accumulate
        dist_ap_sq += tl.sum((a - p) * (a - p), axis=0)
        dist_an_sq += tl.sum((a - n) * (a - n), axis=0)
        
    # Compute Euclidean distances
    dist_ap = tl.sqrt(dist_ap_sq)
    dist_an = tl.sqrt(dist_an_sq)
    
    # Compute the margin loss: max(dist_ap - dist_an + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the result for this sample
    tl.store(out_ptr + row_id, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Wrapper to launch the Triton kernel for Triplet Margin Loss.
    """
    # Ensure tensors are contiguous on GPU
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    # Output tensor to store per-sample loss
    out = torch.empty((batch_size,), device=anchor.device, dtype=anchor.dtype)
    
    # Tunable parameter for feature dimension block size
    BLOCK_SIZE = 1024
    # Grid: one program per sample in the batch
    grid = (batch_size,)
    
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean loss across the batch to match PyTorch's default behavior
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using custom Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace torch.nn.TripletMarginLoss with the optimized Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)