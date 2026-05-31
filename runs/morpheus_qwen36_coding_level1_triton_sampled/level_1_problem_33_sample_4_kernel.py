import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    num_features, N, H, W, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    c = pid
    if c < num_features:
        num_elements = N * H * W
        sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # First pass: compute sum and sum of squares
        for i in range(0, num_elements, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            mask = idx < num_elements
            x_val = tl.load(x_ptr + c * num_elements + idx, mask=mask, other=0.0)
            sum += x_val
            sum_sq += x_val * x_val
        
        # Block reduction
        sum = tl.reduce(sum, 0, tl.sum)
        sum_sq = tl.reduce(sum_sq, 0, tl.sum)
        
        mean = sum / num_elements
        var = sum_sq / num_elements - mean * mean
        var = tl.maximum(var, 0.0)
        
        inv_std = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        
        # Second pass: normalize and scale
        for i in range(0, num_elements, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            mask = idx < num_elements
            x_val = tl.load(x_ptr + c * num_elements + idx, mask=mask, other=0.0)
            out_val = gamma * (x_val - mean) * inv_std + beta
            tl.store(out_ptr + c * num_elements + idx, out_val, mask=mask)


def triton_bn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    grid = lambda meta: (C,)
    bn_kernel[grid](
        x, out, gamma, beta,
        C, N, H, W, eps,
        BLOCK_SIZE=128,
        num_warps=4
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_bn(x, self.gamma, self.beta, self.eps)