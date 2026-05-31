import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * C
    out_ptr += row_idx * C
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # First pass: compute sum and sum of squares
    for start in range(0, C, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq_val += tl.sum(x * x)
        
    mean = sum_val / C
    var = sum_sq_val / C - mean * mean
    inv_std = 1.0 / tl.sqrt(tl.maximum(var + eps, 0.0))
    
    # Second pass: compute output
    for start in range(0, C, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std * w + b
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    assert x.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    N = x.shape[0]
    C = x.numel() // N
    x_flat = x.view(N, C)
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    grid = (N,)
    
    layernorm_kernel[grid](x_flat, weight, bias, out_flat, N, C, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out_flat.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ln.weight
        bias = self.ln.bias
        eps = self.ln.eps
        return triton_layernorm(x, weight, bias, eps)