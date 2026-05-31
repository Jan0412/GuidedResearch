import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def inorm_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    num_spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (batch, channel) pair
    # offsets for spatial dimensions
    offsets = pid * num_spatial + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # First pass: compute mean and variance
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, num_spatial, BLOCK_SIZE):
        block_offsets = start + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < num_spatial
        # Load data for this block
        x = tl.load(x_ptr + pid * num_spatial + block_offsets, mask=block_mask, other=0.0)
        sum_acc += x
        sum_sq_acc += x * x

    # Reduce across threads to get total sum and sum of squares
    total_sum = tl.sum(sum_acc)
    total_sum_sq = tl.sum(sum_sq_acc)

    # Compute mean and variance
    mean = total_sum / num_spatial
    var = total_sum_sq / num_spatial - mean * mean
    std = tl.sqrt(var + eps)

    # Second pass: normalize and store
    for start in range(0, num_spatial, BLOCK_SIZE):
        block_offsets = start + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < num_spatial
        x = tl.load(x_ptr + pid * num_spatial + block_offsets, mask=block_mask, other=0.0)
        out = (x - mean) / std
        tl.store(out_ptr + pid * num_spatial + block_offsets, out, mask=block_mask)


def triton_inorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, height, width = x.shape
    n_elements = x.numel()
    num_spatial = height * width
    
    out = torch.empty_like(x)
    
    # Grid: one program per (batch, channel)
    grid = (batch_size * num_features,)
    BLOCK_SIZE = 128  # Tunable block size
    
    inorm_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        num_spatial=num_spatial,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_inorm(x)