import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    batch_size, dim, margin,
    BLOCK_SIZE
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    anchor_ptr += pid * dim
    pos_ptr += pid * dim
    neg_ptr += pid * dim
    out_ptr += pid

    dist_pos_sq = 0.0
    dist_neg_sq = 0.0

    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim

        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(pos_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(neg_ptr + offsets, mask=mask, other=0.0)

        diff_pos = a - p
        diff_neg = a - n

        dist_pos_sq += tl.sum(diff_pos * diff_pos, axis=0)
        dist_neg_sq += tl.sum(diff_neg * diff_neg, axis=0)

    dist_pos = tl.sqrt(dist_pos_sq)
    dist_neg = tl.sqrt(dist_neg_sq)

    loss = tl.maximum(dist_pos - dist_neg + margin, 0.0)
    tl.store(out_ptr, loss)


def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    batch_size, dim = anchor.shape
    out = torch.empty(batch_size, dtype=torch.float32, device=anchor.device)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        batch_size, dim, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        losses = triton_triplet_loss(anchor, positive, negative, self.margin)
        return torch.mean(losses)