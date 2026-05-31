import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    B, C, H, W, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    bc = pid
    b = bc // C
    c = bc % C
    
    base_ptr = b * C * H * W + c * H * W
    
    sum_val = 0.0
    sum_sq_val = 0.0
    n = H * W
    
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First pass: compute mean and variance over spatial dimensions
    for k in range(num_blocks):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + base_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
        
    mean = sum_val / n
    var = sum_sq_val / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load affine parameters into registers
    g = tl.load(gamma_ptr + c)
    b_val = tl.load(beta_ptr + c)
    
    # Second pass: normalize and apply affine transform
    for k in range(num_blocks):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + base_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std
        out = out * g + b_val
        tl.store(out_ptr + base_ptr + offsets, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (B * C,)
    
    instance_norm_kernel[grid](x, out, gamma, beta, B, C, H, W, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, self.gamma, self.beta)