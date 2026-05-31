import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    N, C, H, W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    x_ptr += n * C * H * W + c * H * W
    out_ptr += n * C * H * W + c * H * W
    
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    
    M1 = 0.0
    M2 = 0.0
    n_count = 0
    
    num_blocks = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block in range(num_blocks):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        count_b = tl.sum(mask)
        if count_b == 0:
            continue
            
        mean_b = tl.sum(x * mask) / count_b
        var_b = tl.sum((x - mean_b)**2 * mask) / count_b
        
        M1 = (M1 * n_count + mean_b * count_b) / (n_count + count_b)
        M2 = M2 + var_b * count_b + (n_count * count_b / (n_count + count_b)) * (M1 - mean_b)**2
        n_count += count_b
        
    var = M2 / n_count
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for block in range(num_blocks):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = gamma * (x - M1) * inv_std + beta
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_instance_norm(x, gamma, beta, eps=1e-05):
    assert x.is_cuda
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    out = torch.empty_like(x)
    
    N, C, H, W = x.shape
    BLOCK_SIZE = 1024
    
    grid = (N * C,)
    
    instance_norm_kernel[grid](x, out, gamma, beta, N, C, H, W, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        return triton_instance_norm(x, self.gamma, self.beta)