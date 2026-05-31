import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    out_ptr,
    margin,
    dim,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row for each input
    a_row_ptr = anchor_ptr + row_idx * dim
    p_row_ptr = pos_ptr + row_idx * dim
    n_row_ptr = neg_ptr + row_idx * dim
    
    dist_p_sq = 0.0
    dist_n_sq = 0.0
    
    # Pre-calculate offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Loop over the dimension in blocks
    for i in range(0, num_blocks):
        curr_offsets = offsets + i * BLOCK_SIZE
        mask = curr_offsets < dim
        
        # Load data
        a = tl.load(a_row_ptr + curr_offsets, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + curr_offsets, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + curr_offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_p = a - p
        diff_n = a - n
        
        # Accumulate squared distances
        dist_p_sq += tl.sum(diff_p * diff_p, axis=0)
        dist_n_sq += tl.sum(diff_n * diff_n, axis=0)
    
    # Compute Euclidean distances
    dist_p = tl.sqrt(dist_p_sq)
    dist_n = tl.sqrt(dist_n_sq)
    
    # Compute triplet loss: max(dist_p - dist_n + margin, 0)
    loss = tl.maximum(dist_p - dist_n + margin, 0.0)
    
    # Store the loss for this sample
    tl.store(out_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    """
    Triton wrapper for Triplet Margin Loss.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    out = torch.empty((batch_size,), device=anchor.device, dtype=anchor.dtype)
    
    BLOCK_SIZE = 1024
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid is one program per sample in the batch
    grid = (batch_size,)
    
    triplet_margin_loss_kernel[grid](
        anchor, 
        positive, 
        negative, 
        out, 
        margin, 
        dim, 
        num_blocks, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # PyTorch TripletMarginLoss defaults to 'mean' reduction
    return out.mean()

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using a fused Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use the custom Triton implementation for fused distance and loss calculation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)