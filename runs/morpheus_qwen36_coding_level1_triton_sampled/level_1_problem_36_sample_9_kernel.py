import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    num_features,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * num_features
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_features
    
    x = tl.load(x_ptr + base_offset, mask=mask, other=0.0)
    
    x2 = x * x
    sum_x2 = tl.sum(x2, axis=0)
    rms = tl.sqrt(sum_x2 / num_features + eps)
    
    out = x / rms
    tl.store(out_ptr + base_offset, out, mask=mask)

def rms_norm_triton(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    out = torch.empty_like(x)
    
    num_features = x.shape[1]
    n_elements = x.numel()
    
    BLOCK_SIZE = 64
    
    grid = (n_elements // num_features,)
    
    rms_norm_kernel[grid](x, out, num_features, eps, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_triton(x, self.eps)