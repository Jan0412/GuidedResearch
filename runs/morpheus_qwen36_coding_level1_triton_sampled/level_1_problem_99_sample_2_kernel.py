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
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    anchor_ptr += batch_idx * dim
    positive_ptr += batch_idx * dim
    negative_ptr += batch_idx * dim

    dist_pos = 0.0
    dist_neg = 0.0

    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim

        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

        diff_pos = a - p
        diff_neg = a - n

        dist_pos += tl.sum(diff_pos * diff_pos, axis=0)
        dist_neg += tl.sum(diff_neg * diff_neg, axis=0)

    loss = tl.maximum(dist_pos - dist_neg + margin, 0.0)
    tl.store(out_ptr + batch_idx, loss)


def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    batch_size = anchor.shape[0]
    dim = anchor.shape[1]
    out = torch.empty(batch_size, dtype=anchor.dtype, device=anchor.device)

    BLOCK_SIZE = 1024
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
        return triton_triplet_loss(anchor, positive, negative, self.margin)