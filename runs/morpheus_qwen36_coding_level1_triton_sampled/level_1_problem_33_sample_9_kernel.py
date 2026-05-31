import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, N, C, H, W, eps, BLOCK_SIZE: tl.constexpr):
    c = tl.program_id(0)
    total = N * H * W
    
    # First pass: compute mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for k in range(num_blocks):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
        
    mean = sum_x / total
    var = sum_x2 / total - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load scale and shift parameters for this channel
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    
    # Second pass: normalize and apply affine transform
    for k in range(num_blocks):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_bn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 128  # Tunable block size
    grid = (C,)
    
    bn_kernel[grid](x, gamma, beta, out, N, C, H, W, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_bn(x, self.bn.weight, self.bn.bias)


def get_inputs():
    x = torch.rand(64, 64, 512, 512)
    return [x]

def get_init_inputs():
    return [64]