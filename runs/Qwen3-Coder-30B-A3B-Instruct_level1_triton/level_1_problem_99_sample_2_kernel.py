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
        
    # Process one sample per thread
    sample_idx = start_idx + tl.thread_id(0)
    
    if sample_idx >= batch_size:
        return
        
    # Calculate pointers for this sample
    anchor_ptr_sample = anchor_ptr + sample_idx * dim
    positive_ptr_sample = positive_ptr + sample_idx * dim
    negative_ptr_sample = negative_ptr + sample_idx * dim
    
    # Compute distances
    pos_dist = 0.0
    neg_dist = 0.0
    
    # Compute L2 distance squared for positive pair
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        anchor_vals = tl.load(anchor_ptr_sample + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr_sample + offsets, mask=mask, other=0.0)
        
        diff = anchor_vals - positive_vals
        pos_dist += tl.sum(diff * diff)
    
    # Compute L2 distance squared for negative pair
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        anchor_vals = tl.load(anchor_ptr_sample + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_ptr_sample + offsets, mask=mask, other=0.0)
        
        diff = anchor_vals - negative_vals
        neg_dist += tl.sum(diff * diff)
    
    # Take square root to get actual distances
    pos_dist = tl.sqrt(pos_dist + eps)
    neg_dist = tl.sqrt(neg_dist + eps)
    
    # Compute loss for this sample
    loss = tl.maximum(0.0, pos_dist - neg_dist + margin)
    
    # Store the result
    tl.store(output_ptr + sample_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of triplet margin loss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA"
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have the same shape"
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    output = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    triplet_margin_loss_kernel[grid_size, 1](
        anchor,
        positive,
        negative,
        output,
        margin,
        2.0,  # p (L2 norm)
        1e-8,  # eps
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return output.mean()

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    Optimized with custom Triton kernels.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use custom Triton kernel instead of PyTorch's built-in TripletMarginLoss
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)