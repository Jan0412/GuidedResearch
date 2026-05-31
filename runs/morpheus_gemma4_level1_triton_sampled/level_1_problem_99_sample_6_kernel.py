import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    out_ptr,
    stride_a_b,
    stride_a_d,
    dim,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to the batch index
    batch_id = tl.program_id(0)
    
    # Pointers for the current batch element
    # Since we call .contiguous() in the wrapper, strides are consistent across tensors
    anchor_row_ptr = anchor_ptr + batch_id * stride_a_b
    pos_row_ptr = pos_ptr + batch_id * stride_a_b
    neg_row_ptr = neg_ptr + batch_id * stride_a_b
    
    dist_ap_sq = 0.0
    dist_an_sq = 0.0
    
    # Loop over the feature dimension in blocks
    for k in range(0, dim, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load elements using the row pointer and offsets multiplied by the dimension stride
        a = tl.load(anchor_row_ptr + offsets * stride_a_d, mask=mask, other=0.0)
        p = tl.load(pos_row_ptr + offsets * stride_a_d, mask=mask, other=0.0)
        n = tl.load(neg_row_ptr + offsets * stride_a_d, mask=mask, other=0.0)
        
        # Compute squared differences and accumulate the sum
        # (a - p)^2 and (a - n)^2
        diff_ap = a - p
        diff_an = a - n
        dist_ap_sq += tl.sum(diff_ap * diff_ap)
        dist_an_sq += tl.sum(diff_an * diff_an)
        
    # Final distance calculation (L2 norm)
    dist_ap = tl.sqrt(dist_ap_sq)
    dist_an = tl.sqrt(dist_an_sq)
    
    # Triplet margin loss formula: L = max(d(a,p) - d(a,n) + margin, 0)
    loss = tl.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Store the result for this batch element
    tl.store(out_ptr + batch_id, loss)

def triton_triplet_margin_loss(anchor, positive, negative, margin=1.0):
    # Ensure inputs are on CUDA
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous to simplify indexing and ensure consistent strides
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    out = torch.empty((batch_size,), device=anchor.device, dtype=anchor.dtype)
    
    # Strides for the tensor (batch stride and dimension stride)
    stride_a_b = anchor.stride(0)
    stride_a_d = anchor.stride(1)
    
    # Tunable block size for the reduction loop
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    # Launch the Triton kernel
    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out,
        stride_a_b, stride_a_d, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # The PyTorch TripletMarginLoss returns the mean of the losses across the batch by default
    return out.mean()

class ModelNew(nn.Module):
    """
    An optimized model that computes Triplet Margin Loss using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace the PyTorch TripletMarginLoss with the custom Triton implementation
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)