import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, G, eps,
    BLOCK_SIZE: tl.constexpr
):
    ng = tl.program_id(0)
    n = ng // G
    g = ng % G
    
    H_W = H * W
    C_per_group = C // G
    
    base = n * C * H_W + g * C_per_group * H_W
    K = C_per_group * H_W
    
    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(num_blocks):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + K
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq_val += tl.sum(x * x)
        
    mean = sum_val / K
    var = sum_sq_val / K - mean * mean
    std = tl.sqrt(var + eps)
    
    for i in range(num_blocks):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + K
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        x_norm = (x - mean) / std
        
        local_offsets = offsets - base
        channel_idx = local_offsets // H_W % C_per_group
        
        w = tl.load(w_ptr + g * C_per_group + channel_idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + g * C_per_group + channel_idx, mask=mask, other=0.0)
        
        out = w * x_norm + b
        tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        x = x.contiguous()
        out = torch.empty_like(x)
        
        N, C, H, W = x.shape
        G = self.num_groups
        
        grid = lambda meta: (N * G,)
        
        group_norm_kernel[grid](
            x, self.weight, self.bias, out,
            N, C, H, W, G, self.eps,
            BLOCK_SIZE=1024,
            num_warps=4
        )
        return out