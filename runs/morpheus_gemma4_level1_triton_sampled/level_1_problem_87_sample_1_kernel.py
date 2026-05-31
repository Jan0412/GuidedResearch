import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointers to the start of the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the input and weight blocks
    # x is (M, K), w is (N, K)
    # We want to perform Y = X @ W.T
    # X block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # W block: (BLOCK_SIZE_K, BLOCK_SIZE_N) -> we load W[n, k] as W[k, n]
    x_block_ptr = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_block_ptr = w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Offset pointers for the current K-block
        curr_x_ptr = x_block_ptr + k * BLOCK_SIZE_K * stride_xk
        curr_w_ptr = w_block_ptr + k * BLOCK_SIZE_K * stride_wk

        # Load blocks with masking
        mask_x = (rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K)
        mask_w = ((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N)
        
        x = tl.load(curr_x_ptr, mask=mask_x, other=0.0)
        w = tl.load(curr_w_ptr, mask=mask_w, other=0.0)

        accumulator += tl.dot(x, w)

    # Add bias
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    accumulator += bias[None, :]

    # Store result
    out_block_ptr = out_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_block_ptr, accumulator, mask=mask_out)

def triton_pointwise_conv(x, weight, bias):
    # x: (B, Cin, H, W)
    # weight: (Cout, Cin, 1, 1)
    # bias: (Cout,)
    B, Cin, H, W = x.shape
    Cout = weight.shape[0]

    # 1. Reshape/Permute input to (B*H*W, Cin) to treat as GEMM
    # NCHW -> NHWC -> (B*H*W, Cin)
    x_permuted = x.permute(0, 2, 3, 1).contiguous()
    M = B * H * W
    K = Cin
    N = Cout

    # 2. Squeeze weights to (Cout, Cin)
    w_squeezed = weight.squeeze(-1).squeeze(-1).contiguous()

    # 3. Prepare output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Triton Parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    pointwise_conv_kernel[grid](
        x_permuted, w_squeezed, bias, out,
        M, N, K,
        x_permuted.stride(0), x_permuted.stride(1),
        w_squeezed.stride(0), w_squeezed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # 4. Reshape output back to (B, Cout, H, W)
    # (B*H*W, Cout) -> (B, H, W, Cout) -> (B, Cout, H, W)
    out = out.view(B, H, W, Cout).permute(0, 3, 1, 2).contiguous()
    return out

class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # We maintain the same parameter shapes as nn.Conv2d for compatibility
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on GPU and FP32
        x = x.cuda().float()
        weight = self.weight.cuda().float()
        bias = self.bias.cuda().float() if self.bias is not None else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        
        return triton_pointwise_conv(x, weight, bias)