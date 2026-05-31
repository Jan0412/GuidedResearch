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
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Each block processes one sample
    if block_idx >= batch_size:
        return
    
    # Calculate offsets for this sample
    anchor_offset = block_idx * dim
    positive_offset = block_idx * dim
    negative_offset = block_idx * dim
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Process all dimensions in chunks
    for i in range(0, dim, BLOCK_SIZE):
        # Create offsets for current chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor, positive, and negative vectors
        anchor_vals = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        # Accumulate squared distances
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Take square root for L2 norm
    dist_ap = tl.sqrt(dist_ap + eps)
    dist_an = tl.sqrt(dist_an + eps)
    
    # Compute loss
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + block_idx, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Triton kernel implementation of triplet margin loss computation
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA"
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have same shape"
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    output = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Constants
    BLOCK_SIZE = 1024
    eps = 1e-8
    
    # Grid configuration
    grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor,
        positive,
        negative,
        output,
        margin,
        2.0,  # L2 norm
        eps,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.mean()  # Return mean loss

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for triplet margin loss computation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use our Triton-based implementation instead of PyTorch's built-in
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)