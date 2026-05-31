import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_ba, stride_bb, stride_bc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    x = tl.program_id(0)
    y = tl.program_id(1)

    offs_m = x * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = y * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_n[:, None] * stride_ba + k + offs_k[None, :]
        b_ptrs = B_ptr + offs_m[:, None] * stride_bb + k + offs_k[None, :]

        mask_a = mask_n[:, None] & mask_k[None, :]
        mask_b = mask_m[:, None] & mask_k[None, :]

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_m
        bias = tl.load(bias_ptrs, mask=mask_m, other=0.0)
        acc += bias[:, None]

    c_ptrs = C_ptr + offs_n[:, None] * stride_bc + offs_m[None, :]
    mask_c = mask_n[:, None] & mask_m[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, bias: torch.Tensor = None):
    M, K = B.shape
    N_batch, K_a = A.shape
    assert K == K_a, f"K mismatch: {K} vs {K_a}"

    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    stride_ba = K
    stride_bb = K
    stride_bc = M

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N_batch, BLOCK_N)

    grid = (num_blocks_m, num_blocks_n)

    HAS_BIAS = bias is not None

    batched_matmul_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C, bias_ptr=bias,
        M=M, N=N_batch, K=K,
        stride_ba=stride_ba, stride_bb=stride_bb, stride_bc=stride_bc,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_BIAS=HAS_BIAS,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32)) if bias else None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        N_batch = B * H * W
        
        x_reshaped = x.reshape(N_batch, C_in).contiguous()
        w = self.weight
        
        out_reshaped = torch.empty(N_batch, self.out_channels, dtype=torch.float32, device=x.device)
        
        bias = self.bias if self.bias is not None else None
        
        out_reshaped = triton_batched_matmul(x_reshaped, w, out_reshaped, bias)
        
        out = out_reshaped.reshape(B, self.out_channels, H, W)
        return out