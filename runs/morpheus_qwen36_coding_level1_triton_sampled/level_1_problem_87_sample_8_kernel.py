import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_ak = offs_k < K
    
    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_ak_k = offs_k < (K - k)
        
        a = tl.load(A_ptrs, mask=mask_am[:, None] & mask_ak_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_bn[None, :] & mask_ak_k[:, None], other=0.0)
        
        accumulator += tl.dot(a, b)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=mask_bn, other=0.0)
        accumulator += bias
        
    C_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(C_ptrs, accumulator, mask=mask_am[:, None] & mask_bn[None, :])


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    M = B * H * W
    N = C_out
    K = C_in
    
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)
    
    HAS_BIAS = bias is not None
    
    gemm_bias_kernel[grid](
        x, weight, out, bias if HAS_BIAS else None,
        M, N, K,
        K, 1,
        N, 1,
        N, 1,
        BLOCK_M, BLOCK_N, BLOCK_K,
        HAS_BIAS,
    )
    
    out = out.view(B, C_out, H, W)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias)