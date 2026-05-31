import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, 
    positive_ptr, 
    negative_ptr, 
    out_ptr, 
    margin, 
    D, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one triplet in the batch
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row for each input
    anchor_row_ptr = anchor_ptr + row_idx * D
    positive_row_ptr = positive_ptr + row_idx * D
    negative_row_ptr = negative_ptr + row_idx * D
    
    dist_ap_sq = 0.0
    dist_an_sq = 0.0
    
    # Iterate over the feature dimension in blocks
    for i in range(0, D, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        
        # Load feature blocks
        a = tl.load(anchor_row_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(positive_row_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(negative_row_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate squared Euclidean distances
        dist_ap_sq += tl.sum((a - p) * (a - p))
        dist_an_sq += tl.sum((a - n) * (a - n))
    
    # Compute final distance (L2 norm)
    dist_ap = tl.sqrt(dist_ap_sq)
    dist_an = tl.sqrt(dist_an_sq)
    
    # Compute triplet margin loss: max(dist_ap - dist_an + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the per-sample loss
    tl.store(out_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0):
    """
    Triton wrapper for Triplet Margin Loss.
    Computes the mean loss across the batch.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    out = torch.empty((batch_size,), device=anchor.device, dtype=anchor.dtype)
    
    # Tuning parameter for the dimension loop
    BLOCK_SIZE = 1024
    
    # Grid is one program per triplet in the batch
    grid = (batch_size,)
    
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out,
        margin,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # PyTorch's TripletMarginLoss defaults to 'mean' reduction
    return out.mean()

class ModelNew(nn.Module):
    """
    An optimized model that computes Triplet Margin Loss using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use the fused Triton kernel instead of nn.TripletMarginLoss
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)