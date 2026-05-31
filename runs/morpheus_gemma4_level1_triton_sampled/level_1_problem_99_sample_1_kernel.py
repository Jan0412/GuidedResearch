import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr, out_ptr,
    N, D, margin,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute the per-sample Triplet Margin Loss.
    Each program handles one sample in the batch.
    """
    # Row index (sample index in the batch)
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return

    # Base pointers for the current sample's vectors
    a_row_ptr = anchor_ptr + row_idx * D
    p_row_ptr = positive_ptr + row_idx * D
    n_row_ptr = negative_ptr + row_idx * D

    sum_ap = 0.0
    sum_an = 0.0

    # Process the vector dimension D in chunks of BLOCK_SIZE
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        
        # Load input values for anchor, positive, and negative samples
        a = tl.load(a_row_ptr + cols, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + cols, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + cols, mask=mask, other=0.0)
        
        # Compute squared Euclidean distances
        diff_ap = a - p
        diff_an = a - n
        
        # Accumulate the squared differences
        sum_ap += tl.sum(diff_ap * diff_ap, axis=0)
        sum_an += tl.sum(diff_an * diff_an, axis=0)

    # Compute L2 norms with a small epsilon for numerical stability
    dist_ap = tl.sqrt(sum_ap + 1e-6)
    dist_an = tl.sqrt(sum_an + 1e-6)
    
    # Triplet Margin Loss: L = max(d(a, p) - d(a, n) + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the resulting loss for the current sample
    tl.store(out_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Wrapper function to launch the Triton kernel for Triplet Margin Loss.
    """
    # Ensure inputs are contiguous and on the same CUDA device
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    N, D = anchor.shape
    # Output tensor to store per-sample losses
    out = torch.empty((N,), device=anchor.device, dtype=anchor.dtype)
    
    # Tuning parameter: block size for the vector dimension reduction
    BLOCK_SIZE = 1024
    # Grid dimension: one program per sample in the batch
    grid = (N,)
    
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out,
        N, D, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean loss over the batch, matching PyTorch's default reduction='mean'
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace torch.nn.TripletMarginLoss with the fused Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)