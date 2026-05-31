import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr, out_ptr,
    N, M, K,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // K
    k = pid % K
    
    acc = 0.0
    for m in range(0, M, BLOCK_SIZE_M):
        offsets_m = m + tl.arange(0, BLOCK_SIZE_M)
        mask = offsets_m < M
        ptr = x_ptr + n * M * K + offsets_m * K + k
        x_block = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(x_block)
    
    tl.store(out_ptr + n * K + k, acc)


def triton_sum_reduce(x: torch.Tensor, dim: int):
    assert x.is_cuda and dim == 1, "Currently optimized for dim=1 on CUDA."
    x = x.contiguous()
    N, M, K = x.shape
    out = torch.empty((N, 1, K), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 128
    
    grid = (N, K)
    sum_reduce_kernel[grid](x, out, N, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)