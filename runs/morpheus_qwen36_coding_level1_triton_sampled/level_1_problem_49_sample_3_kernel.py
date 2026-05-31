import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr, out_ptr, stride_x0, stride_x1, stride_x2,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    max_val = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], float('-inf'), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + offsets_k
        x = tl.load(x_ptr + offsets_m[:, None] * stride_x0 + offsets_n[None, :] * stride_x1 + k_offsets[None, :] * stride_x2,
                    mask=mask_2d & (k_offsets[None, :] < K), other=float('-inf'))
        max_val = tl.maximum(max_val, x)
        
    tl.store(out_ptr + offsets_m[:, None] * stride_x0 + offsets_n[None, :] * stride_x1, max_val, mask=mask_2d)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    
    if dim != 2:
        perm = list(range(x.dim()))
        perm.remove(dim)
        perm.append(dim)
        x = x.permute(perm).contiguous()
        
    M, N, K = x.shape
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, 1)
    
    max_reduce_kernel[grid](
        x, out, N*K, K, 1,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    if dim != 2:
        out = out.permute(perm).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)