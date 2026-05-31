import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm_stats_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    stride_c,
    block_size_c,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_c
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < block_size_c

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(0, block_size_c, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + i
        mask = offsets < block_size_c
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        sum += x
        sum_sq += x * x

    mean = tl.sum(sum) / block_size_c
    var = tl.sum(sum_sq) / block_size_c - mean * mean
    var = tl.maximum(var, eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def batchnorm_forward_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    H,
    W,
    C,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < n_elements

    c = (offsets // (H * W)) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + c, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0)

    out = (x - mean) / tl.sqrt(var + eps) * gamma + beta

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_batchnorm_stats(x: torch.Tensor):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    B, C, H, W = x.shape
    stride_c = x.stride(1)
    block_size_c = B * H * W
    n_elements = x.numel()

    mean = torch.empty(C, dtype=x.dtype, device=x.device)
    var = torch.empty(C, dtype=x.dtype, device=x.device)

    grid = (C,)
    batchnorm_stats_kernel[grid](
        x, mean, var, stride_c, block_size_c, n_elements, 1e-5, BLOCK_SIZE=1024
    )
    return mean, var


def triton_batchnorm_forward(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    B, C, H, W = x.shape
    n_elements = x.numel()

    out = torch.empty_like(x)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    batchnorm_forward_kernel[grid](
        x, mean, var, gamma, beta, out, H, W, C, n_elements, 1e-5, BLOCK_SIZE=128
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.randn(num_features))
        self.beta = nn.Parameter(torch.randn(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean, var = triton_batchnorm_stats(x)
        else:
            mean = self.running_mean
            var = self.running_var
        return triton_batchnorm_forward(x, mean, var, self.gamma, self.beta)