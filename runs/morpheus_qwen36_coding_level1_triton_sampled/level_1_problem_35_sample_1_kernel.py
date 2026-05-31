import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    eps,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    # Base offset for the current batch and group
    base_offset = b * C * H * W + g * K * H * W

    # Accumulators for mean and variance
    total_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Pass 1: Compute sum and sum of squares
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(num_blocks):
        block_start = block_id * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offsets) < N

        # Compute channel, height, width from linear index within group
        c_rel = block_start // (H * W)
        rem = block_start % (H * W)
        h = rem // W
        w = rem % W

        x_offset = base_offset + c_rel * H * W + h * W + w
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

        block_sum = tl.reduce(x, axis=0, op=tl.add)
        total_sum += block_sum

        block_sum_sq = tl.reduce(x * x, axis=0, op=tl.add)
        total_sum_sq += block_sum_sq

    # Compute mean and variance
    mean = total_sum / N
    var = total_sum_sq / N - mean * mean
    rsqrt_var = tl.math.rsqrt(var + eps)

    # Pass 2: Normalize and apply affine parameters
    for block_id in range(num_blocks):
        block_start = block_id * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offsets) < N

        c_rel = block_start // (H * W)
        rem = block_start % (H * W)
        h = rem // W
        w = rem % W

        x_offset = base_offset + c_rel * H * W + h * W + w
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

        diff = x - mean
        norm_x = diff * rsqrt_var

        weight_val = tl.load(weight_ptr + g * K + c_rel)
        bias_val = tl.load(bias_ptr + g * K + c_rel)

        out = norm_x * weight_val + bias_val
        tl.store(out_ptr + x_offset, out, mask=mask)


def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    B, C, H, W = x.shape
    G = weight.numel() // C
    K = C // G
    N = K * H * W
    eps = 1e-5
    BLOCK_SIZE = 128

    out = torch.empty_like(x)
    grid = (B, G)

    group_norm_kernel[grid](
        x, weight, bias, out,
        B, N, eps, C, H, W, K, G, BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_group_norm(x, self.gn.weight, self.gn.bias)