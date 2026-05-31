import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    out_ptr,
    n_features,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    anchor_ptr += batch_idx * n_features
    positive_ptr += batch_idx * n_features
    negative_ptr += batch_idx * n_features
    out_ptr += batch_idx

    sum_ap = 0.0
    sum_an = 0.0

    num_blocks = (n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

        diff_ap = a - p
        diff_an = a - n
        sum_ap += tl.sum(diff_ap * diff_ap)
        sum_an += tl.sum(diff_an * diff_an)

    loss = tl.maximum(0.0, sum_ap - sum_an + margin)
    tl.store(out_ptr, loss)


def triton_triplet_loss(anchor, positive, negative, margin):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    batch_size = anchor.shape[0]
    n_features = anchor.shape[1]
    out = torch.empty(batch_size, dtype=torch.float32, device=anchor.device)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        n_features, margin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)