import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    a_ptr, p_ptr, n_ptr, out_ptr,
    batch_size, dim, margin,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    sum_ap = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_an = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for off in range(0, dim, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        a_vals = tl.load(a_ptr + pid * dim + offsets, mask=mask, other=0.0)
        p_vals = tl.load(p_ptr + pid * dim + offsets, mask=mask, other=0.0)
        n_vals = tl.load(n_ptr + pid * dim + offsets, mask=mask, other=0.0)

        diff_ap = a_vals - p_vals
        diff_an = a_vals - n_vals

        sum_ap += diff_ap * diff_ap
        sum_an += diff_an * diff_an

    loss = tl.sum(sum_ap) - tl.sum(sum_an) + margin
    loss = tl.maximum(loss, 0.0)
    tl.store(out_ptr + pid, loss)


def triton_triplet_loss(a, p, n, margin):
    assert a.is_cuda and p.is_cuda and n.is_cuda
    a = a.contiguous()
    p = p.contiguous()
    n = n.contiguous()

    batch_size = a.shape[0]
    dim = a.shape[1]
    out = torch.empty(batch_size, dtype=torch.float32, device=a.device)

    grid = (batch_size,)
    triplet_loss_kernel[grid](a, p, n, out, batch_size, dim, margin, BLOCK_SIZE=128)
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)