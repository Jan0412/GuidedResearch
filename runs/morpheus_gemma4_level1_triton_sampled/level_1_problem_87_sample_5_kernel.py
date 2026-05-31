import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Ranges for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to X and W
    # X is (M, K), W is (K, N)
    x_ptr_base = x_ptr
    w_ptr_base = w_ptr

    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load X block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # offset = row * stride_xm + col * stride_xk
        x_offsets = rm[:, None] * stride_xm + (k * BLOCK_SIZE_K + rk[None, :]) * stride_xk
        x = tl.load(x_ptr + x_offsets, mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K), other=0.0)

        # Load W block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # offset = row * stride_wk + col * stride_wn
        w_offsets = (k * BLOCK_SIZE_K + rk[:, None]) * stride_wk + rn[None, :] * stride_wn
        w = tl.load(w_ptr + w_offsets, mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication
        acc += tl.dot(x, w)

    # Add bias: Bias is (N,)
    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc += bias[None, :]

    # Store result: (BLOCK_SIZE_M, BLOCK_SIZE_N)
    out_offsets = rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptr + out_offsets, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_pointwise_conv(x, weight, bias):
    """
    Triton wrapper for 1x1 convolution.
    x: (B, Cin, H, W)
    weight: (Cout, Cin, 1, 1)
    bias: (Cout,)
    """
    B, Cin, H, W = x.shape
    Cout = weight.shape[0]

    # 1. Reshape x to (B*H*W, Cin)
    # permute(0, 2, 3, 1) -> (B, H, W, Cin) -> view(-1, Cin)
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, Cin)
    
    # 2. Reshape weight to (Cin, Cout)
    # permute(1, 0, 2, 3) -> (Cin, Cout, 1, 1) -> view(Cin, Cout)
    w_flat = weight.permute(1, 0, 2, 3).reshape(Cin, Cout).contiguous()

    M = x_flat.shape[0] # B * H * W
    K = x_flat.shape[1] # Cin
    N = w_flat.shape[1] # Cout

    # Prepare output tensor
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Handle bias
    if bias is not None:
        b_flat = bias.contiguous()
    else:
        b_flat = torch.zeros(N, device=x.device, dtype=torch.float32)

    # Hyperparameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Strides
    stride_xm, stride_xk = x_flat.stride()
    stride_wk, stride_wn = w_flat.stride()
    stride_om, stride_on = out_flat.stride()

    conv1x1_kernel[grid](
        x_flat, w_flat, b_flat, out_flat,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # 3. Reshape output back to (B, Cout, H, W)
    # out_flat: (B*H*W, Cout) -> view(B, H, W, Cout) -> permute(0, 3, 1, 2)
    return out_flat.view(B, H, W, Cout).permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to maintain parameters and initialization
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Ensure inputs are FP32 and on CUDA
        x = x.to(torch.float32)
        weight = self.conv1d.weight.to(torch.float32)
        bias = self.conv1d.bias.to(torch.float32) if self.conv1d.bias is not None else None
        
        return triton_pointwise_conv(x, weight, bias)