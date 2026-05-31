import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_kernel(
    anchor_ptr, positive_ptr, negative_ptr, out_ptr,
    batch_size, dim, margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Pointers to the start of the rows for the current sample
    a_row_ptr = anchor_ptr + pid * dim
    p_row_ptr = positive_ptr + pid * dim
    n_row_ptr = negative_ptr + pid * dim

    dist_ap_sq = 0.0
    dist_an_sq = 0.0

    # Loop over the dimension in blocks to compute the squared L2 norm
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load elements for anchor, positive, and negative
        a = tl.load(a_row_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared differences and accumulate sums
        dist_ap_sq += tl.sum((a - p) * (a - p), axis=0)
        dist_an_sq += tl.sum((a - n) * (a - n), axis=0)

    # Compute L2 distances
    dist_ap = tl.sqrt(dist_ap_sq)
    dist_an = tl.sqrt(dist_an_sq)
    
    # Compute triplet margin loss: max(dist_ap - dist_an + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the per-sample loss
    tl.store(out_ptr + pid, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Triton-optimized implementation of TripletMarginLoss.
    Fuses distance calculations and margin application into a single kernel.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for efficient pointer arithmetic
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    out = torch.empty((batch_size,), device=anchor.device, dtype=anchor.dtype)
    
    # BLOCK_SIZE is the number of elements processed per iteration for the distance sum
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    triplet_margin_kernel[grid](
        anchor, positive, negative, out,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean loss across the batch to match PyTorch's default reduction
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model computing Triplet Margin Loss using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace torch.nn.TripletMarginLoss with the fused Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)