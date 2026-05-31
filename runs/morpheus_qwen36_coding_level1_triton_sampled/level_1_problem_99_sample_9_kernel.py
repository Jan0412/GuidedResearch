import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    a_ptr, p_ptr, n_ptr, out_ptr,
    batch_size, dim,
    margin,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    d_pos = 0.0
    d_neg = 0.0

    for off in range(0, dim, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + offsets, mask=mask, other=0.0)

        d_pos += tl.sum((a - p) * (a - p))
        d_neg += tl.sum((a - n) * (a - n))

    loss = tl.maximum(d_pos - d_neg + margin, 0.0)
    tl.store(out_ptr + pid, loss)


def triton_triplet_loss(anchor, positive, negative, margin):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    assert anchor.shape == positive.shape == negative.shape

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
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2
    )
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)