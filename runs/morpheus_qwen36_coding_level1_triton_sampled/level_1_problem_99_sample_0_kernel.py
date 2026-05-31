import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    out_ptr,
    M, N, margin,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * N
    sum_pos = 0.0
    sum_neg = 0.0

    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N

        a = tl.load(anchor_ptr + row_offset + offsets, mask=mask, other=0.0)
        p = tl.load(positive_ptr + row_offset + offsets, mask=mask, other=0.0)
        n = tl.load(negative_ptr + row_offset + offsets, mask=mask, other=0.0)

        diff_pos = a - p
        diff_neg = a - n

        sum_pos += tl.sum(diff_pos * diff_pos)
        sum_neg += tl.sum(diff_neg * diff_neg)

    loss = tl.maximum(sum_pos - sum_neg + margin, 0.0)
    tl.store(out_ptr + pid, loss)


def triton_triplet_loss(anchor, positive, negative, margin):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    M, N = anchor.shape
    out = torch.empty(M, device=anchor.device, dtype=torch.float32)

    BLOCK_N = 1024
    grid = (M,)

    triplet_loss_kernel[grid](
        anchor, positive, negative, out,
        M, N, margin,
        BLOCK_N=BLOCK_N
    )
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)