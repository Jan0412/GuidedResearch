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
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one sample in the batch
    if pid >= batch_size:
        return
        
    # Calculate offsets for this sample
    anchor_offset = pid * dim
    positive_offset = pid * dim
    negative_offset = pid * dim
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Compute L2 distance components
    for i in range(0, dim, BLOCK_SIZE):
        # Create offsets for current block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor, positive, negative vectors
        anchor_vals = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        # Accumulate squared distances
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Take square root
    dist_ap = tl.sqrt(dist_ap + eps)
    dist_an = tl.sqrt(dist_an + eps)
    
    # Compute loss for this sample
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + pid, loss)

@triton.jit
def fused_triplet_loss_forward_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    margin,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one sample in the batch
    if pid >= batch_size:
        return
        
    # Calculate offsets for this sample
    anchor_offset = pid * dim
    positive_offset = pid * dim
    negative_offset = pid * dim
    
    # Compute distances using vectorized operations
    dist_ap = 0.0
    dist_an = 0.0
    
    # Process in blocks
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load vectors
        anchor_vals = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        # Accumulate squared distances
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Compute final loss
    dist_ap = tl.sqrt(dist_ap)
    dist_an = tl.sqrt(dist_an)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + pid, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Custom Triton implementation of triplet margin loss
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA"
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have same shape"
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    losses = torch.empty(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Determine grid size
    grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    fused_triplet_loss_forward_kernel[grid](
        anchor, 
        positive, 
        negative, 
        losses,
        margin,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return losses.mean()

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for triplet margin loss computation
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)

# Keep the original functions for compatibility
batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape), torch.rand(batch_size, *input_shape)]
    
def get_init_inputs():
    return [1.0]  # Default margin