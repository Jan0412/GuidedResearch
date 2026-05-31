import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr, out_ptr,
    stride_a, stride_p, stride_n,
    n_rows, dim, margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample (row) in the batch
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Pointers to the start of the row for anchor, positive, and negative tensors
    a_row_ptr = anchor_ptr + row_idx * stride_a
    p_row_ptr = positive_ptr + row_idx * stride_p
    n_row_ptr = negative_ptr + row_idx * stride_n

    dist_ap_sq = 0.0
    dist_an_sq = 0.0

    # Process the dimension in blocks to handle large feature vectors (e.g., 8192)
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load values for the current block
        a = tl.load(a_row_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared differences for the current block
        diff_p = a - p
        diff_n = a - n
        
        # Accumulate the sum of squares
        dist_ap_sq += tl.sum(diff_p * diff_p)
        dist_an_sq += tl.sum(diff_n * diff_n)

    # Compute L2 distances (Euclidean)
    dist_ap = tl.sqrt(dist_ap_sq)
    dist_an = tl.sqrt(dist_an_sq)
    
    # Compute Triplet Margin Loss: max(d(a, p) - d(a, n) + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the computed loss for this sample
    tl.store(out_ptr + row_idx, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """
    Wrapper function to launch the Triton kernel for Triplet Margin Loss.
    """
    # Ensure inputs are contiguous and on GPU
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    n_rows, dim = anchor.shape
    # Output tensor to hold the loss per sample
    out = torch.empty((n_rows,), device=anchor.device, dtype=torch.float32)
    
    # Get strides for row-wise indexing
    stride_a = anchor.stride(0)
    stride_p = positive.stride(0)
    stride_n = negative.stride(0)
    
    # Tune BLOCK_SIZE based on the dimension; 1024 is efficient for 8192
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        stride_a, stride_p, stride_n,
        n_rows, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the mean of the loss across the batch to match PyTorch's default behavior
    return torch.mean(out)


class ModelNew(nn.Module):
    """
    Optimized Model using a custom Triton kernel for Triplet Margin Loss.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Use the fused Triton kernel instead of torch.nn.TripletMarginLoss
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)