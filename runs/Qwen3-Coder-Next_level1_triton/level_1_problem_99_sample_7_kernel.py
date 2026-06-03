import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    batch_size,
    dim,
    margin,
    eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each block processes one sample in the batch
    batch_idx = tl.program_id(0)
    
    # Calculate pointers for this batch
    anchor_offset = batch_idx * dim
    positive_offset = batch_idx * dim
    negative_offset = batch_idx * dim
    
    # Accumulators for squared distances
    dist_a_p = 0.0
    dist_a_n = 0.0
    
    # Process in blocks to compute squared distances
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor, positive, negative values
        a = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_a_p = a - p
        diff_a_n = a - n
        
        dist_a_p += tl.sum(diff_a_p * diff_a_p)
        dist_a_n += tl.sum(diff_a_n * diff_a_n)
    
    # Add epsilon for numerical stability
    dist_a_p = tl.sqrt(dist_a_p + eps)
    dist_a_n = tl.sqrt(dist_a_n + eps)
    
    # Compute triplet margin loss
    loss = tl.maximum(dist_a_p - dist_a_n + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, reduction='mean'):
    """
    Custom Triton implementation of TripletMarginLoss for FP32 tensors.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "Input tensors must have the same shape."
    
    batch_size, dim = anchor.shape[0], anchor.shape[1]
    
    # Ensure contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Set block size based on dimension
    BLOCK_SIZE = min(256, dim)
    NUM_BLOCKS = triton.cdiv(dim, BLOCK_SIZE)
    
    # Grid: one block per batch sample
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, output,
        batch_size, dim, margin, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BLOCKS=NUM_BLOCKS
    )
    
    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:  # 'none'
        return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for TripletMarginLoss computation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin)