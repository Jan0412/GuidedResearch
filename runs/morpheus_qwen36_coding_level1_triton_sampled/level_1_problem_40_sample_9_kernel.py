import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr += pid * N
    out_ptr += pid * N
    
    # Pass 1: Compute Mean
    sum_val = tl.zeros([BLOCK_SIZE_N], dtype=tl.float64)
    for start in range(0, N, BLOCK_SIZE_N):
        offsets = start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.where(mask, x.to(tl.float64), 0.0)
    mean = tl.sum(sum_val) / N
    
    # Pass 2: Compute Variance
    var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float64)
    for start in range(0, N, BLOCK_SIZE_N):
        offsets = start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        var += tl.where(mask, (x.to(tl.float64) - mean) ** 2, 0.0)
    var = tl.sum(var) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Pass 3: Apply Scale, Shift and Store Output
    for start in range(0, N, BLOCK_SIZE_N):
        offsets = start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std * w + b
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size = x.shape[0]
    N = x.numel() // batch_size
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE_N = 1024
    grid = (batch_size,)
    
    layernorm_kernel[grid](x, weight, bias, out, N, eps, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_layernorm(x, self.weight, self.bias, self.eps)


def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    features = 64
    dim1 = 256
    dim2 = 256
    return [(features, dim1, dim2)]