import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    batch_size, feature_size,
    margin,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one batch element
    batch_id = tl.program_id(0)
    
    # Offset to the start of this batch element's data
    base_offset = batch_id * feature_size
    
    # Accumulator for positive distance (anchor - positive)^p
    pos_dist = tl.zeros((1,), dtype=tl.float32)
    # Accumulator for negative distance (anchor - negative)^p
    neg_dist = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate through features in blocks
    for start in range(0, feature_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < feature_size
        
        # Load values
        a = tl.load(anchor_ptr + base_offset + offsets, mask=mask, other=0.0)
        pos = tl.load(positive_ptr + base_offset + offsets, mask=mask, other=0.0)
        neg = tl.load(negative_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        # Compute differences
        diff_pos = a - pos
        diff_neg = a - neg
        
        # Compute absolute values raised to power p (for p=2, it's squared difference)
        if p == 2:
            pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
            neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
        else:
            # For general p, use abs(diff)^p
            abs_diff_pos = tl.abs(diff_pos)
            abs_diff_neg = tl.abs(diff_neg)
            pos_dist += tl.sum(tl.pow(abs_diff_pos, p), axis=0)
            neg_dist += tl.sum(tl.pow(abs_diff_neg, p), axis=0)
    
    # Compute final loss: max(pos_dist - neg_dist + margin, 0)
    loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
    
    # Store result (for batched input, we return mean loss)
    tl.store(output_ptr, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2):
    """
    Triton implementation of TripletMarginLoss for FP32 tensors.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "Input shapes must match."
    
    batch_size = anchor.shape[0]
    feature_size = anchor.shape[1] if len(anchor.shape) > 1 else 1
    
    # For simplicity, we'll compute per-batch and then average
    # But we can optimize by computing all in one kernel
    # Output tensor for the mean loss
    output = torch.zeros(1, device=anchor.device, dtype=torch.float32)
    
    # Use appropriate block size based on feature dimension
    BLOCK_SIZE = min(256, feature_size)
    
    # Launch kernel for each batch element (could be optimized further with fusion)
    grid = (batch_size,)
    
    # Compute per-batch losses
    per_batch_losses = torch.zeros(batch_size, device=anchor.device, dtype=torch.float32)
    
    # Actually, let's implement a more efficient version that computes the final mean directly
    # Reset and use a single kernel that accumulates the final mean
    output = torch.zeros(1, device=anchor.device, dtype=torch.float32)
    
    # Redesign for single kernel: accumulate sum and then divide by batch_size
    # This requires atomic operations or separate kernel for reduction
    # For simplicity and performance, we'll use a two-stage approach:
    # 1. Compute per-batch loss
    # 2. Reduce to mean
    
    # Let's optimize with a single fused kernel that computes the mean directly
    # Create temporary storage for per-batch losses
    per_batch = torch.empty(batch_size, device=anchor.device, dtype=torch.float32)
    
    # Launch kernel for each batch element
    for b in range(batch_size):
        # Compute this batch element's loss
        # Use a small grid for feature dimension
        feature_block_size = min(256, feature_size)
        
        # Compute the squared distances for this batch element
        pos_sum = 0.0
        neg_sum = 0.0
        
        # For simplicity, implement a fused kernel that computes the result
        pass
    
    # Instead, implement a more efficient approach with a single kernel
    @triton.jit
    def triplet_margin_loss_fused_kernel(
        anchor_ptr, positive_ptr, negative_ptr,
        output_ptr,
        batch_size, feature_size,
        margin,
        p: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        # Accumulator for total loss
        total_loss = tl.zeros((1,), dtype=tl.float32)
        
        # Process batches in parallel
        batch_start = tl.program_id(0) * BLOCK_SIZE
        for offset in range(0, batch_size, BLOCK_SIZE):
            batch_ids = batch_start + tl.arange(0, BLOCK_SIZE)
            mask = batch_ids < batch_size
            
            # For each batch element, compute distances
            for b in range(BLOCK_SIZE):
                if batch_ids[b] < batch_size:
                    bid = batch_ids[b]
                    base_offset = bid * feature_size
                    
                    # Accumulate for this batch element
                    pos_dist = tl.zeros((1,), dtype=tl.float32)
                    neg_dist = tl.zeros((1,), dtype=tl.float32)
                    
                    for f in range(0, feature_size, BLOCK_SIZE):
                        offsets = f + tl.arange(0, BLOCK_SIZE)
                        feat_mask = offsets < feature_size
                        
                        a = tl.load(anchor_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                        pos = tl.load(positive_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                        neg = tl.load(negative_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                        
                        diff_pos = a - pos
                        diff_neg = a - neg
                        
                        if p == 2:
                            pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
                            neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
                        else:
                            abs_diff_pos = tl.abs(diff_pos)
                            abs_diff_neg = tl.abs(diff_neg)
                            pos_dist += tl.sum(tl.pow(abs_diff_pos, p), axis=0)
                            neg_dist += tl.sum(tl.pow(abs_diff_neg, p), axis=0)
                    
                    # Compute loss for this batch element
                    batch_loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
                    total_loss += batch_loss
        
        # Store mean loss
        mean_loss = total_loss / batch_size
        tl.store(output_ptr, mean_loss)
    
    # Use a simpler approach - compute per-batch with vectorized operations
    # The key insight is to fuse the subtraction and distance computation
    
    # Let's implement a more practical version
    @triton.jit
    def triplet_loss_per_batch_kernel(
        anchor_ptr, positive_ptr, negative_ptr,
        output_ptr,
        batch_idx,
        feature_size,
        margin,
        p: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        # Compute for a single batch element
        base_offset = batch_idx * feature_size
        
        pos_dist = tl.zeros((1,), dtype=tl.float32)
        neg_dist = tl.zeros((1,), dtype=tl.float32)
        
        for start in range(0, feature_size, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < feature_size
            
            a = tl.load(anchor_ptr + base_offset + offsets, mask=mask, other=0.0)
            pos = tl.load(positive_ptr + base_offset + offsets, mask=mask, other=0.0)
            neg = tl.load(negative_ptr + base_offset + offsets, mask=mask, other=0.0)
            
            diff_pos = a - pos
            diff_neg = a - neg
            
            if p == 2:
                pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
                neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
            else:
                pos_dist += tl.sum(tl.pow(tl.abs(diff_pos), p), axis=0)
                neg_dist += tl.sum(tl.pow(tl.abs(diff_neg), p), axis=0)
        
        # Compute loss for this batch element
        loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
        
        # Store result
        tl.store(output_ptr + batch_idx, loss)
    
    # Actually, let's create a more optimized fused kernel
    @triton.jit
    def triplet_margin_loss_optimized_kernel(
        anchor_ptr, positive_ptr, negative_ptr,
        output_ptr,
        batch_size, feature_size,
        margin,
        p: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        # Each program handles a subset of batches
        batch_start = tl.program_id(0) * BLOCK_SIZE
        num_programs = tl.num_programs(0)
        
        # Accumulator for this program's portion
        total_loss = tl.zeros((1,), dtype=tl.float32)
        count = 0
        
        # Process batches in this program's range
        for b in range(BLOCK_SIZE):
            bid = batch_start + b
            if bid >= batch_size:
                break
                
            base_offset = bid * feature_size
            pos_dist = tl.zeros((1,), dtype=tl.float32)
            neg_dist = tl.zeros((1,), dtype=tl.float32)
            
            # Process features in blocks
            for f_start in range(0, feature_size, BLOCK_SIZE):
                offsets = f_start + tl.arange(0, BLOCK_SIZE)
                feat_mask = offsets < feature_size
                
                # Load anchor, positive, negative values
                a = tl.load(anchor_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                pos = tl.load(positive_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                neg = tl.load(negative_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
                
                # Compute differences
                diff_pos = a - pos
                diff_neg = a - neg
                
                # Compute distances based on p-norm
                if p == 2:
                    pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
                    neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
                else:
                    pos_dist += tl.sum(tl.pow(tl.abs(diff_pos), p), axis=0)
                    neg_dist += tl.sum(tl.pow(tl.abs(diff_neg), p), axis=0)
            
            # Compute loss for this batch element
            batch_loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
            total_loss += batch_loss
            count += 1
        
        # Store partial sum
        tl.atomic_add(output_ptr, total_loss)
        # For simplicity, we'll do the mean calculation in PyTorch
    
    # Use the optimized approach with a simpler kernel
    @triton.jit
    def triplet_margin_loss_final_kernel(
        anchor_ptr, positive_ptr, negative_ptr,
        output_ptr,
        batch_size, feature_size,
        margin,
        p: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        # Each program handles one batch element
        batch_idx = tl.program_id(0)
        
        # Base offset for this batch element
        base_offset = batch_idx * feature_size
        
        # Accumulators
        pos_dist = tl.zeros((1,), dtype=tl.float32)
        neg_dist = tl.zeros((1,), dtype=tl.float32)
        
        # Process features in blocks
        for f_start in range(0, feature_size, BLOCK_SIZE):
            offsets = f_start + tl.arange(0, BLOCK_SIZE)
            feat_mask = offsets < feature_size
            
            # Load values
            a = tl.load(anchor_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
            pos = tl.load(positive_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
            neg = tl.load(negative_ptr + base_offset + offsets, mask=feat_mask, other=0.0)
            
            # Compute differences
            diff_pos = a - pos
            diff_neg = a - neg
            
            # Compute distances
            if p == 2:
                pos_dist += tl.sum(diff_pos * diff_pos, axis=0)
                neg_dist += tl.sum(diff_neg * diff_neg, axis=0)
            else:
                pos_dist += tl.sum(tl.pow(tl.abs(diff_pos), p), axis=0)
                neg_dist += tl.sum(tl.pow(tl.abs(diff_neg), p), axis=0)
        
        # Compute loss
        loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)
        
        # Store result
        tl.store(output_ptr + batch_idx, loss)
    
    # Prepare output
    per_batch_losses = torch.empty(batch_size, device=anchor.device, dtype=torch.float32)
    
    # Calculate grid size
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_final_kernel[grid](
        anchor, positive, negative,
        per_batch_losses,
        batch_size, feature_size,
        margin,
        p=p,
        BLOCK_SIZE=256
    )
    
    # Return mean loss
    return per_batch_losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for TripletMarginLoss computation.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Call our custom Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin)