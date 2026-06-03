import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    batch_size, dim,
    margin,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one batch element
    batch_id = tl.program_id(0)
    
    # Offset to the start of this batch element
    anchor_offset = batch_id * dim
    positive_offset = batch_id * dim
    negative_offset = batch_id * dim
    
    # Accumulators for squared distances
    dist_p_sq = 0.0
    dist_n_sq = 0.0
    
    # Compute squared distances in chunks
    num_blocks = tl.cdiv(dim, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load anchor, positive, negative values
        a = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_p = a - p
        diff_n = a - n
        dist_p_sq += tl.sum(diff_p * diff_p, axis=0)
        dist_n_sq += tl.sum(diff_n * diff_n, axis=0)
    
    # Compute distances (square roots)
    dist_p = tl.sqrt(dist_p_sq)
    dist_n = tl.sqrt(dist_n_sq)
    
    # Compute triplet loss: max(dist_p - dist_n + margin, 0)
    loss = tl.maximum(dist_p - dist_n + margin, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_id, loss)


def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0):
    """
    Compute triplet margin loss using Triton kernel.
    
    Args:
        anchor: Tensor of shape (batch_size, dim)
        positive: Tensor of shape (batch_size, dim)
        negative: Tensor of shape (batch_size, dim)
        margin: Margin for the loss
        
    Returns:
        Scalar tensor with the mean loss over the batch
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    
    # Output tensor for individual losses
    losses = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Launch kernel - one block per batch element
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    triplet_loss_kernel[grid](
        anchor, positive, negative, losses,
        batch_size, dim,
        margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using Triton kernels.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)