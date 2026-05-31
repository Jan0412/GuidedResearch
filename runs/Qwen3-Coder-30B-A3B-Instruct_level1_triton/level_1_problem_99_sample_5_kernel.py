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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load anchor, positive, and negative embeddings
    anchor = tl.load(anchor_ptr + offsets[:, None] * dim + tl.arange(0, dim)[None, :], mask=mask[:, None], other=0.0)
    positive = tl.load(positive_ptr + offsets[:, None] * dim + tl.arange(0, dim)[None, :], mask=mask[:, None], other=0.0)
    negative = tl.load(negative_ptr + offsets[:, None] * dim + tl.arange(0, dim)[None, :], mask=mask[:, None], other=0.0)
    
    # Compute distances
    # Distance to positive
    diff_pos = anchor - positive
    dist_pos = tl.sqrt(tl.sum(diff_pos * diff_pos, axis=1) + eps)
    
    # Distance to negative
    diff_neg = anchor - negative
    dist_neg = tl.sqrt(tl.sum(diff_neg * diff_neg, axis=1) + eps)
    
    # Compute loss
    loss = tl.maximum(dist_pos - dist_neg + margin, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, loss, mask=mask)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6):
    """
    Triton implementation of Triplet Margin Loss computation
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "All tensors must have the same shape."
    
    batch_size, dim = anchor.shape
    
    # Prepare output tensor
    output = torch.zeros(batch_size, dtype=torch.float32, device=anchor.device)
    
    # Kernel parameters
    BLOCK_SIZE = 128
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
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
    
    return output.mean()

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using Triton optimizations.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use the Triton kernel for computing the loss
        loss = triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin)
        return loss

# Helper functions for the Triton kernel
def get_inputs():
    scale = torch.rand(())
    return [torch.rand(32768, 8192)*scale, torch.rand(32768, 8192), torch.rand(32768, 8192)]
    
def get_init_inputs():
    return [1.0]  # Default margin