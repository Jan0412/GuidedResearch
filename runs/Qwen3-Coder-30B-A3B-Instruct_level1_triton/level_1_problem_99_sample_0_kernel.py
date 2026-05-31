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
        
    # Process one sample at a time
    for i in range(start_idx, min(start_idx + BLOCK_SIZE, batch_size)):
        # Compute distance between anchor and positive
        dist_ap = 0.0
        for j in range(dim):
            diff = tl.load(anchor_ptr + i * dim + j) - tl.load(positive_ptr + i * dim + j)
            dist_ap += diff * diff
            
        # Compute distance between anchor and negative
        dist_an = 0.0
        for j in range(dim):
            diff = tl.load(anchor_ptr + i * dim + j) - tl.load(negative_ptr + i * dim + j)
            dist_an += diff * diff
            
        # Apply margin and compute loss
        loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
        
        # Store the result
        tl.store(output_ptr + i, loss)

@triton.jit
def fused_triplet_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    margin,
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
        
    # Process multiple samples per thread if needed
    for i in range(start_idx, min(start_idx + BLOCK_SIZE, batch_size)):
        # Compute distance between anchor and positive
        dist_ap = 0.0
        for j in range(dim):
            diff = tl.load(anchor_ptr + i * dim + j) - tl.load(positive_ptr + i * dim + j)
            dist_ap += diff * diff
            
        # Compute distance between anchor and negative
        dist_an = 0.0
        for j in range(dim):
            diff = tl.load(anchor_ptr + i * dim + j) - tl.load(negative_ptr + i * dim + j)
            dist_an += diff * diff
            
        # Apply margin and compute loss
        loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
        
        # Store the result
        tl.store(output_ptr + i, loss)

def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    """
    Computes triplet margin loss using Triton kernel
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have the same shape."
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1]
    
    # Ensure tensors are contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    # Prepare output tensor
    output = torch.zeros(batch_size, dtype=torch.float32, device='cuda')
    
    # Kernel parameters
    BLOCK_SIZE = 128
    grid = (math.ceil(batch_size / BLOCK_SIZE),)
    
    # Launch the Triton kernel
    fused_triplet_loss_kernel[grid](
        anchor,
        positive,
        negative,
        output,
        margin,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean of all losses
    return output.mean()

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    Optimized with custom Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)