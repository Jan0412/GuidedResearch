import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr, out_ptr,
    N, K, M,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // M
    m = pid % M
    
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    mask_k = offsets_k < K
    
    stride_n = K * M
    stride_k = M
    stride_m = 1
    
    base_ptr = x_ptr + n * stride_n + m * stride_m
    x_vals = tl.load(base_ptr + offsets_k * stride_k, mask=mask_k, other=-float('inf'))
    
    out_val = tl.max(x_vals, axis=0)
    tl.store(out_ptr + n * M + m, out_val)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    N, K, M = x.shape
    assert dim == 1, "Optimized for dim=1"
    
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_K = 4096
    grid = (N * M,)
    
    max_reduce_kernel[grid](x, out, N, K, M, BLOCK_SIZE_K=BLOCK_SIZE_K)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)