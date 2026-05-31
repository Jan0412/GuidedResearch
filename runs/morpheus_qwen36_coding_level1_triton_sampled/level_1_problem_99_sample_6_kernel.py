import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr, out_ptr,
    batch_size, dim, margin,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * dim
    
    acc_p = 0.0
    acc_n = 0.0
    
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        anchor = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        positive = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
        negative = tl.load(negative_ptr + offsets, mask=mask, other=0.0)
        
        diff_p = anchor - positive
        diff_n = anchor - negative
        
        acc_p += tl.sum(diff_p * diff_p)
        acc_n += tl.sum(diff_n * diff_n)
        
    loss = tl.maximum(acc_p - acc_n + margin, 0.0)
    tl.store(out_ptr + pid, loss)


def triton_triplet_loss(anchor, positive, negative, margin):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size, dim = anchor.shape
    out = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)