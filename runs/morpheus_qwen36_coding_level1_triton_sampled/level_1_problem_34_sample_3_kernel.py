import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    N,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    x_ptr += b * C * H * W + c * H * W
    out_ptr += b * C * H * W + c * H * W

    # Pass 1: Compute sum and sum of squares for mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize and store results
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be contiguous FP32 CUDA tensor."
    x = x.contiguous()

    B, C, H, W = x.shape
    N = H * W
    out = torch.empty_like(x)

    BLOCK_SIZE = 128  # Tunable block size for optimal occupancy
    grid = (B * C,)

    instance_norm_kernel[grid](
        x, out,
        B, C, H, W,
        N,
        BLOCK_SIZE,
        eps=eps
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x)