import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, H, W, C_in, C_out,
    stride_x, stride_w, stride_b, stride_o,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offset < (B * H * W)
    mask_n = n_offset < C_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, C_in, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        mask_k = k_offset < C_in

        x_idx = m_offset[:, None] * stride_x + k_offset[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        w_idx = n_offset[:, None] * stride_w + k_offset[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        w = tl.load(weight_ptr + w_idx, mask=w_mask, other=0.0)

        acc += tl.dot(x, w, allow_tf32=False)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n_offset, mask=mask_n, other=0.0)
        acc += bias[None, :]

    out_idx = m_offset[:, None] * stride_o + n_offset[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, H, W = x.shape
    C_out = weight.shape[0]

    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 4

    grid = (
        (B * H * W + BLOCK_M - 1) // BLOCK_M,
        (C_out + BLOCK_N - 1) // BLOCK_N,
    )

    conv1d_kernel[grid](
        x, weight, bias, out,
        B, H, W, C_in, C_out,
        C_in, C_in, 1, C_out,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_param if self.bias else None
        return triton_conv1d(x, self.weight, bias)


def get_inputs():
    batch_size = 16
    in_channels = 64
    height = 1024
    width = 1024
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]


def get_init_inputs():
    in_channels = 64
    out_channels = 128
    return [in_channels, out_channels]