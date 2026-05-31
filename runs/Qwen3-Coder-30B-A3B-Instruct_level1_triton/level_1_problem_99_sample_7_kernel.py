import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    margin,
    p,
    eps,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate the starting index for this block
    start_idx = block_id * BLOCK_SIZE
    
    # Check if we're within bounds
    if start_idx >= batch_size:
        return
        
    # Process multiple rows per block if needed
    for i in range(BLOCK_SIZE):
        row_idx = start_idx + i
        if row_idx >= batch_size:
            break
            
        # Compute distances using vectorized operations
        # Distance to positive sample
        pos_dist = 0.0
        neg_dist = 0.0
        
        # Compute squared L2 distances
        for d in range(dim):
            anchor_val = tl.load(anchor_ptr + row_idx * dim + d)
            positive_val = tl.load(positive_ptr + row_idx * dim + d)
            negative_val = tl.load(negative_ptr + row_idx * dim + d)
            
            diff_pos = anchor_val - positive_val
            diff_neg = anchor_val - negative_val
            
            pos_dist += diff_pos * diff_pos
            neg_dist += diff_neg * diff_neg
            
        # Apply sqrt for L2 distance
        pos_dist = tl.sqrt(pos_dist + eps)
        neg_dist = tl.sqrt(neg_dist + eps)
        
        # Compute triplet loss
        loss = tl.maximum(0.0, pos_dist - neg_dist + margin)
        
        # Store result
        tl.store(output_ptr + row_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6):
    """
    Custom Triton implementation of Triplet Margin Loss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA"
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have the same shape"
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    output = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor,
        positive,
        negative,
        output,
        margin,
        p,
        eps,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean of all losses
    return torch.mean(output)

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    Optimized with custom Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)