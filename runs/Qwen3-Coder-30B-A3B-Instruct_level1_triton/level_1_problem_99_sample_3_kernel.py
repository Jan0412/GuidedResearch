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
    
    # Calculate the starting indices for this block
    start_idx = block_id * BLOCK_SIZE
    
    # Create a mask for valid elements
    mask = tl.arange(0, BLOCK_SIZE) < batch_size - start_idx
    
    # Initialize loss accumulator
    loss_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process each sample in the batch
    for i in range(BLOCK_SIZE):
        if start_idx + i >= batch_size:
            break
            
        # Compute distances for this sample
        anchor = tl.load(anchor_ptr + (start_idx + i) * dim + tl.arange(0, dim), mask=mask & (tl.arange(0, dim) < dim))
        positive = tl.load(positive_ptr + (start_idx + i) * dim + tl.arange(0, dim), mask=mask & (tl.arange(0, dim) < dim))
        negative = tl.load(negative_ptr + (start_idx + i) * dim + tl.arange(0, dim), mask=mask & (tl.arange(0, dim) < dim))
        
        # Compute L2 distances
        diff_pos = anchor - positive
        diff_neg = anchor - negative
        
        dist_pos = tl.sum(diff_pos * diff_pos)
        dist_neg = tl.sum(diff_neg * diff_neg)
        
        # Apply margin and compute loss
        loss = tl.maximum(dist_neg - dist_pos + margin, 0.0)
        loss_sum[i] = loss
        
    # Store the results
    tl.store(output_ptr + start_idx + tl.arange(0, BLOCK_SIZE), loss_sum, mask=mask)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Triton implementation of triplet margin loss computation
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA"
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1]
    
    # Prepare output tensor
    output = torch.zeros(batch_size, device=anchor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Determine the number of blocks needed
    grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
    
    # Launch the Triton kernel
    triplet_margin_loss_kernel[grid](
        anchor,
        positive,
        negative,
        output,
        margin,
        2.0,  # p value for L2 norm
        1e-8,  # eps value
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss across batch
    return torch.mean(output)

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
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)