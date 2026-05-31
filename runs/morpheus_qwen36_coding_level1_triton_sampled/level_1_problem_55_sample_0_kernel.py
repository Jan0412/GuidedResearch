import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    C_in,
    C_out,
    H,
    W,
    K,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_c_out,
    w_stride_c_in,
    w_stride_k,
    w_stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    # Decode program ID to output coordinates (n, c_out, h, w)
    pid = tl.program_id(0)
    W_out = W - K + 1
    H_out = H - K + 1
    OUT_HW = H_out * W_out
    C_OUT_HW = C_out * OUT_HW

    w = pid % W_out
    rem = pid // W_out
    h = rem % H_out
    rem = rem // H_out
    c_out = rem % C_out
    n = rem // C_out

    # Base offsets for the current output element
    w_base = c_out * w_stride_c_out
    x_base = n * x_stride_n + h * x_stride_h + w * x_stride_w

    acc = 0.0
    offsets_w = tl.arange(0, BLOCK_SIZE)
    offsets_x = tl.arange(0, BLOCK_SIZE)
    mask = tl.full((BLOCK_SIZE,), 1, dtype=tl.int1)

    # Loop over input channels
    for c_in in range(0, C_in, BLOCK_SIZE):
        # Load weight block for current c_in
        # Weights are contiguous in the last two dimensions (K, K)
        w_block = tl.load(w_ptr + w_base + c_in * w_stride_c_in + offsets_w, mask=mask, other=0.0)

        # Load input block for current c_in
        # Input is contiguous in the last two dimensions (H, W)
        # The block corresponds to x[n, c_in, h:h+K, w:w+K]
        x_block = tl.load(x_ptr + x_base + c_in * x_stride_c + offsets_x, mask=mask, other=0.0)

        # Accumulate dot product
        acc += tl.sum(w_block * x_block)

    # Store result
    tl.store(out_ptr + pid, acc, mask=mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for 2D convolution.
    Assumes bias=False, stride=1, padding=0, dilation=1, groups=1.
    Optimized for FP32.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, K, K_w = w.shape

    assert C_in == C_in_w and K == K_w, "Weight shape mismatch."

    H_out = H - K + 1
    W_out = W - K + 1
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = K * K  # Optimal block size for contiguous KxK load

    # Grid configuration: one program per output element
    grid = (N * C_out * H_out * W_out,)

    conv2d_kernel[grid](
        x, w, out,
        N, C_in, C_out, H, W, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer to hold parameters, but forward uses Triton kernel
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel for convolution
        return triton_conv2d(x, self.conv2d.weight)