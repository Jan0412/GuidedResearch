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
    # Get the program ID
    pid = tl.program_id(0)
    
    # Each program handles one sample in the batch
    if pid >= batch_size:
        return
    
    # Calculate offsets for this sample
    anchor_offset = pid * dim
    positive_offset = pid * dim
    negative_offset = pid * dim
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Compute L2 distances using vectorized operations
    for i in range(0, dim, BLOCK_SIZE):
        # Create offsets for current block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load vectors for anchor-positive pair
        anchor_vals = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_ap = anchor_vals - positive_vals
        dist_ap += tl.sum(diff_ap * diff_ap)
        
        # Load vectors for anchor-negative pair
        negative_vals = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        diff_an = anchor_vals - negative_vals
        dist_an += tl.sum(diff_an * diff_an)
    
    # Take square root for L2 distance
    dist_ap = tl.sqrt(dist_ap + eps)
    dist_an = tl.sqrt(dist_an + eps)
    
    # Compute loss for this sample
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the result
    tl.store(output_ptr + pid, loss)

@triton.jit
def triplet_margin_loss_backward_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    grad_output_ptr,
    anchor_grad_ptr,
    positive_grad_ptr,
    negative_grad_ptr,
    margin,
    p,
    eps,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Each program handles one sample in the batch
    if pid >= batch_size:
        return
    
    # Calculate offsets for this sample
    anchor_offset = pid * dim
    positive_offset = pid * dim
    negative_offset = pid * dim
    grad_offset = pid
    
    # Compute distances
    dist_ap = 0.0
    dist_an = 0.0
    
    # Compute L2 distances
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
        
        dist_ap += tl.sum(diff_ap * diff_ap)
        dist_an += tl.sum(diff_an * diff_an)
    
    # Take square root
    dist_ap = tl.sqrt(dist_ap + eps)
    dist_an = tl.sqrt(dist_an + eps)
    
    # Compute gradient multiplier
    grad_mult = 0.0
    if dist_ap - dist_an + margin > 0:
        grad_mult = tl.load(grad_output_ptr + grad_offset)
    
    # Compute gradients
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load vectors
        anchor_vals = tl.load(anchor_ptr + anchor_offset + offsets, mask=mask, other=0.0)
        positive_vals = tl.load(positive_ptr + positive_offset + offsets, mask=mask, other=0.0)
        negative_vals = tl.load(negative_ptr + negative_offset + offsets, mask=mask, other=0.0)
        
        # Compute gradients
        diff_ap = anchor_vals - positive_vals
        diff_an = anchor_vals - negative_vals
        
        # Normalize gradients
        if dist_ap > 0:
            grad_ap = diff_ap / (dist_ap + eps)
        else:
            grad_ap = 0.0
            
        if dist_an > 0:
            grad_an = diff_an / (dist_an + eps)
        else:
            grad_an = 0.0
            
        # Apply gradient multiplier
        grad_ap = grad_mult * grad_ap
        grad_an = grad_mult * grad_an
        
        # Store gradients
        tl.store(anchor_grad_ptr + anchor_offset + offsets, grad_ap - grad_an, mask=mask)
        tl.store(positive_grad_ptr + positive_offset + offsets, -grad_ap, mask=mask)
        tl.store(negative_grad_ptr + negative_offset + offsets, grad_an, mask=mask)

class TritonTripletMarginLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor, positive, negative, margin=1.0, p=2, eps=1e-6):
        ctx.save_for_backward(anchor, positive, negative)
        ctx.margin = margin
        ctx.p = p
        ctx.eps = eps
        
        batch_size = anchor.size(0)
        dim = anchor.size(1)
        
        # Prepare output
        output = torch.empty(batch_size, dtype=torch.float32, device=anchor.device)
        
        # Kernel launch parameters
        BLOCK_SIZE = 128
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
        
        return output.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        anchor, positive, negative = ctx.saved_tensors
        margin = ctx.margin
        p = ctx.p
        eps = ctx.eps
        
        batch_size = anchor.size(0)
        dim = anchor.size(1)
        
        # Initialize gradients
        anchor_grad = torch.zeros_like(anchor)
        positive_grad = torch.zeros_like(positive)
        negative_grad = torch.zeros_like(negative)
        
        # Kernel launch parameters
        BLOCK_SIZE = 128
        grid = lambda meta: (math.ceil(batch_size / meta["BLOCK_SIZE"]),)
        
        # Launch backward kernel
        triplet_margin_loss_backward_kernel[grid](
            anchor,
            positive,
            negative,
            grad_output,
            anchor_grad,
            positive_grad,
            negative_grad,
            margin,
            p,
            eps,
            batch_size,
            dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return anchor_grad, positive_grad, negative_grad, None, None, None

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    Optimized with custom Triton kernels.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use custom Triton implementation instead of PyTorch's built-in
        return TritonTripletMarginLoss.apply(anchor, positive, negative, self.margin)