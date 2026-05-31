import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, pos_ptr, neg_ptr, loss_ptr,
    dim_size, margin,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    base_offset = pid * dim_size

    sum_pos = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_neg = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, dim_size, BLOCK_SIZE):
        offsets_chunk = start + offsets
        mask = offsets_chunk < dim_size

        a = tl.load(anchor_ptr + base_offset + offsets_chunk, mask=mask, other=0.0)
        p = tl.load(pos_ptr + base_offset + offsets_chunk, mask=mask, other=0.0)
        n = tl.load(neg_ptr + base_offset + offsets_chunk, mask=mask, other=0.0)

        diff_pos = a - p
        diff_neg = a - n

        sum_pos += diff_pos * diff_pos
        sum_neg += diff_neg * diff_neg

    dist_pos = tl.sqrt(tl.sum(sum_pos))
    dist_neg = tl.sqrt(tl.sum(sum_neg))

    diff_loss = dist_pos - dist_neg + margin
    loss = tl.where(diff_loss > 0.0, diff_loss, 0.0)
    tl.store(loss_ptr + pid, loss)


def triton_triplet_loss(anchor, pos, neg, margin):
    assert anchor.is_cuda and pos.is_cuda and neg.is_cuda
    anchor = anchor.contiguous()
    pos = pos.contiguous()
    neg = neg.contiguous()

    batch_size = anchor.shape[0]
    dim_size = anchor.shape[1]
    loss = torch.empty(batch_size, dtype=torch.float32, device=anchor.device)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    triplet_loss_kernel[grid](
        anchor, pos, neg, loss,
        dim_size, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return loss.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)