import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, N, eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    x_ptr += pid * N
    out_ptr += pid * N
    
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Pass 1: Compute mean
    sum_val = 0.0
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
    mean = sum_val / N

    # Pass 2: Compute variance
    sum_sq = 0.0
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_centered = x - mean
        sum_sq += tl.sum(x_centered * x_centered)
    var = sum_sq / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: Scale and shift
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = w * (x - mean) * rstd + b
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    B = x.shape[0]
    N = x.numel() // B
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 4096
    grid = (B,)
    
    layer_norm_kernel[grid](x, weight, bias, out, B, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x):
        return triton_layer_norm(x, self.ln.weight, self.ln.bias, self.ln.eps)