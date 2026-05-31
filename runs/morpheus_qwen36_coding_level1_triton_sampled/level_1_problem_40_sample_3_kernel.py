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
    batch_size,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_ptr = x_ptr + pid * N
    
    # Pass 1: Compute mean
    sum_val = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(base_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
    mean = sum_val / N
    
    # Pass 2: Compute variance and apply normalization
    var = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(base_ptr + offsets, mask=mask, other=0.0)
        var += tl.sum((x - mean) ** 2)
    var = var / N
    
    # Pass 3: Scale and shift
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(base_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) / tl.sqrt(var + eps)
        out = out * w + b
        tl.store(out_ptr + pid * N + offsets, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size = x.shape[0]
    N = x.numel() // batch_size
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    
    layernorm_kernel[grid](
        x, out, weight, bias, batch_size, N, eps, BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_layernorm(x, self.weight, self.bias)