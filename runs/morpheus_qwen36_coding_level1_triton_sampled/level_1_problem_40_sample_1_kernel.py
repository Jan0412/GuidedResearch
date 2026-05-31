import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr += pid * C
    out_ptr += pid * C

    # Accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # First pass: compute sum and sum of squares
    for start in range(0, C, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += x
        sum_sq_val += x * x

    # Reduce sums to scalars
    mean = tl.sum(sum_val) / C
    var = tl.sum(sum_sq_val) / C - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply weight and bias
    for start in range(0, C, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    B = x.shape[0]
    C = x.numel() // B
    BLOCK_SIZE = 128
    grid = (B,)
    layernorm_kernel[grid](x, out, weight, bias, B, C, eps, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.C = 1
        for dim in normalized_shape:
            self.C *= dim
        self.weight = nn.Parameter(torch.ones(self.C))
        self.bias = nn.Parameter(torch.zeros(self.C))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_layernorm(x, self.weight, self.bias, self.eps)