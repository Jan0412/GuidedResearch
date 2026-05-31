import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
    eps: float = 1e-5
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    offset = n * C * H * W + c * H * W
    num_elements = H * W
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # First pass: compute sum and sum of squares
    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    
    mean = sum_val / num_elements
    var = sum_sq_val / num_elements - mean * mean
    std = tl.sqrt(var + eps)
    
    # Second pass: normalize
    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
        out = (x - mean) / std
        tl.store(out_ptr + offset + offsets, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    N, C, H, W = x.shape
    grid = (N * C,)
    BLOCK_SIZE = 1024
    instance_norm_kernel[grid](x, out, N, C, H, W, BLOCK_SIZE=BLOCK_SIZE, eps=eps)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, eps=1e-5)