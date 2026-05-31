import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    A_ptr, P_ptr, N_ptr, margin,
    out_ptr,
    batch_size, dim,
    BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return
        
    offsets_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offsets_d < dim
    
    acc_ap = 0.0
    acc_an = 0.0
    
    num_blocks = (dim + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    for i in range(num_blocks):
        offsets_d = i * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offsets_d < dim
        
        a = tl.load(A_ptr + pid * dim + offsets_d, mask=mask_d, other=0.0)
        p = tl.load(P_ptr + pid * dim + offsets_d, mask=mask_d, other=0.0)
        n = tl.load(N_ptr + pid * dim + offsets_d, mask=mask_d, other=0.0)
        
        diff_ap = a - p
        diff_an = a - n
        
        acc_ap += tl.sum(diff_ap * diff_ap, axis=0)
        acc_an += tl.sum(diff_an * diff_an, axis=0)
        
    loss = tl.maximum(0.0, acc_ap - acc_an + margin)
    tl.store(out_ptr + pid, loss)


def triton_triplet_loss(anchor, positive, negative, margin):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()
    
    batch_size = anchor.shape[0]
    dim = anchor.shape[1]
    out = torch.empty(batch_size, dtype=anchor.dtype, device=anchor.device)
    
    BLOCK_SIZE_D = 1024
    grid = (batch_size,)
    
    triplet_loss_kernel[grid](
        anchor, positive, negative, margin,
        out,
        batch_size, dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)