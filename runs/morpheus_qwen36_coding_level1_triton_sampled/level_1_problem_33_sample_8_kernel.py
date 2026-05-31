import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_inference_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr += pid * N
    out_ptr += pid * N
    w = tl.load(weight_ptr + pid)
    b = tl.load(bias_ptr + pid)
    mu = tl.load(running_mean_ptr + pid)
    var = tl.load(running_var_ptr + pid)
    inv_std = 1.0 / tl.sqrt(var + eps)
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = (x - mu) * inv_std * w + b
        tl.store(out_ptr + offsets, y, mask=mask)


def triton_bn(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda and running_mean.is_cuda and running_var.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    C = x.shape[1]
    N = x.numel() // C
    BLOCK_SIZE = 256
    grid = (C,)
    bn_inference_kernel[grid](x, running_mean, running_var, weight, bias, out, C, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.bn.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_bn(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps)