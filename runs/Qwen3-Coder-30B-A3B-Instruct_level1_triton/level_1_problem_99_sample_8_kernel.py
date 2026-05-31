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
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    # Calculate pointers for this sample
    anchor_base = anchor_ptr + batch_idx * dim
    positive_base = positive_ptr + batch_idx * dim
    negative_base = negative_ptr + batch_idx * dim
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Compute L2 distances
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        anchor_vals = tl.load(anchor_base + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_base + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_base + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Take square root for actual distances
    dist_ap = tl.sqrt(dist_ap + eps)
    dist_an = tl.sqrt(dist_an + eps)
    
    # Compute loss for this sample
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)

@triton.jit
def compute_distances_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    dist_ap_ptr,
    dist_an_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    # Calculate pointers for this sample
    anchor_base = anchor_ptr + batch_idx * dim
    positive_base = positive_ptr + batch_idx * dim
    negative_base = negative_ptr + batch_idx * dim
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Compute L2 distances
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        anchor_vals = tl.load(anchor_base + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_base + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_base + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Store results
    tl.store(dist_ap_ptr + batch_idx, tl.sqrt(dist_ap + 1e-8))
    tl.store(dist_an_ptr + batch_idx, tl.sqrt(dist_an + 1e-8))

@triton.jit
def compute_triplet_loss_kernel(
    dist_ap_ptr,
    dist_an_ptr,
    output_ptr,
    margin,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    # Load distances
    dist_ap = tl.load(dist_ap_ptr + batch_idx)
    dist_an = tl.load(dist_an_ptr + batch_idx)
    
    # Compute loss for this sample
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of Triplet Margin Loss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have same shape."
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    losses = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Configure block size
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
    
    # Launch kernel to compute distances
    dist_ap = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    dist_an = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Compute distances using a single kernel
    compute_distances_kernel[grid](anchor, positive, negative, dist_ap, dist_an, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute final losses
    compute_triplet_loss_kernel[grid](dist_ap, dist_an, losses, margin, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return mean loss
    return losses.mean()

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Triplet Margin Loss computation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)