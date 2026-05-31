import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
    x2 = x * x
    sum_x2 = tl.sum(x2)
    rms = tl.sqrt(sum_x2 / n_elements + eps)
    out = x / rms
    tl.store(out_ptr + pid * n_elements + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    
    B, F, D1, D2 = x.shape
    N = D1 * D2
    x_flat = x.view(B * F, N)
    out_flat = out.view(B * F, N)
    
    BLOCK_SIZE = N
    
    grid = (B * F,)
    rms_norm_kernel[grid](x_flat, out_flat, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)