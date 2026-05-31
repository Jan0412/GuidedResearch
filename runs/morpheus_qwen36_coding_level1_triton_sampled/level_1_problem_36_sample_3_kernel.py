import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    N, F, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_offset = row_idx * F

    # First pass: compute sum of squares
    sum_sq = 0.0
    for start_offset in range(0, F, BLOCK_SIZE):
        offsets = base_offset + start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base_offset + F
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean_sq = sum_sq / F
    rms = tl.sqrt(mean_sq + eps)

    # Second pass: normalize
    for start_offset in range(0, F, BLOCK_SIZE):
        offsets = base_offset + start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base_offset + F
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x / rms, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    N = x.numel() // x.shape[1]
    F = x.shape[1]

    BLOCK_SIZE = 128

    grid = (N,)
    rms_norm_kernel[grid](x, out, N, F, eps, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)


def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]