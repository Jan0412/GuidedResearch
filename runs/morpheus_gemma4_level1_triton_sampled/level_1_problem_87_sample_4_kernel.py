import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    has_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers for X (M, K) and W (K, N)
    # We assume BLOCK_K == K for this pointwise operation given the constraints
    # x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    # w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # Load X block
    x = tl.load(
        x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, 
        mask=(rm[:, None] < M) & (rk[None, :] < K), 
        other=0.0
    )
    
    # Load W block
    w = tl.load(
        w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn, 
        mask=(rk[:, None] < K) & (rn[None, :] < N), 
        other=0.0
    )

    # Perform matrix multiplication
    out = tl.dot(x, w)

    # Add bias if present
    if has_bias:
        bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
        out += bias[None, :]

    # Store result
    tl.store(
        out_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, 
        out, 
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )

def triton_pointwise_conv(x, weight, bias=None):
    """
    Wrapper for the Triton pointwise convolution kernel.
    x: (B, Cin, H, W)
    weight: (Cout, Cin, 1, 1)
    bias: (Cout,)
    """
    B, Cin, H, W = x.shape
    Cout = weight.shape[0]
    
    # Reshape and permute input to (M, K) where M = B*H*W, K = Cin
    # NCHW -> NHWC
    x_permuted = x.permute(0, 2, 3, 1).contiguous()
    M = B * H * W
    K = Cin
    N = Cout

    # Reshape and transpose weight to (K, N)
    w_transposed = weight.view(Cout, Cin).t().contiguous()
    
    # Prepare output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Strides
    stride_xm, stride_xk = x_permuted.stride()
    stride_wk, stride_wn = w_transposed.stride()
    stride_ym, stride_yn = out.stride()
    
    # Kernel constants
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = K # Since Cin is small (64), we can process it in one block
    
    # Ensure BLOCK_K is a power of 2 for tl.dot (pad if necessary, but 64 is power of 2)
    # If K were not a power of 2, we would need to pad or loop.
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    has_bias = bias is not None
    b_ptr = bias if has_bias else None

    pointwise_conv_kernel[grid](
        x_permuted, w_transposed, b_ptr, out,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    # Reshape output back to (B, Cout, H, W)
    # (M, N) -> (B, H, W, N) -> (B, N, H, W)
    out = out.view(B, H, W, N).permute(0, 3, 1, 2).contiguous()
    return out

class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to manage parameters (weights and bias)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Ensure input is FP32
        x = x.float()
        weight = self.conv1d.weight.float()
        bias = self.conv1d.bias.float() if self.conv1d.bias is not None else None
        
        return triton_pointwise_conv(x, weight, bias)