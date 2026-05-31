import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,
    im2col_ptr,
    B, C, D, H, W,
    Kd, Kh, Kw,
    Sd, Sh, Sw,
    Pd, Ph, Pw,
    Dd, Dh, Dw,
    out_d, out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_patches = out_d * out_h * out_w
    batch_idx = pid // num_patches
    patch_idx = pid % num_patches

    d_out = patch_idx // (out_h * out_w)
    h_out = (patch_idx // out_w) % out_h
    w_out = patch_idx % out_w

    d_start = d_out * Sd - Pd
    h_start = h_out * Sh - Ph
    w_start = w_out * Sw - Pw

    offsets_kd = tl.arange(0, Kd)
    offsets_kh = tl.arange(0, Kh)
    offsets_kw = tl.arange(0, Kw)
    offsets_c = tl.arange(0, C)

    d_coords = d_start + offsets_kd * Dd
    h_coords = h_start + offsets_kh * Dh
    w_coords = w_start + offsets_kw * Dw

    d_mask = (d_coords >= 0) & (d_coords < D)
    h_mask = (h_coords >= 0) & (h_coords < H)
    w_mask = (w_coords >= 0) & (w_coords < W)
    c_mask = offsets_c < C

    # Broadcast masks for loading
    d_mask = d_mask[:, None, None]
    h_mask = h_mask[None, :, None]
    w_mask = w_mask[None, None, :]
    c_mask = c_mask[None, None, None]

    mask = d_mask & h_mask & w_mask & c_mask

    # Load input
    x_offsets = (
        batch_idx * C * D * H * W +
        offsets_c[:, None, None, None] * D * H * W +
        d_coords[None, :, None, None] * H * W +
        h_coords[None, None, :, None] * W +
        w_coords[None, None, None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    # Store to im2col matrix
    # im2col shape: (B * num_patches, C * Kd * Kh * Kw)
    # Each thread block writes one row
    row_start = pid * C * Kd * Kh * Kw
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < C * Kd * Kh * Kw

    # Reshape x to 1D for storage
    x_flat = x.flatten()
    tl.store(im2col_ptr + row_start + col_offsets, x_flat[col_offsets], mask=col_mask)


@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        B = tl.load(B_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(A, B)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store output
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Register weights and bias as buffers or parameters
        self.register_buffer('weight', torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.register_buffer('bias', torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        Kd, Kh, Kw = self.kernel_size
        Sd, Sh, Sw = self.stride
        Pd, Ph, Pw = self.padding
        Dd, Dh, Dw = self.dilation

        # Calculate output dimensions
        out_d = (D + 2 * Pd - Dd * (Kd - 1) - 1) // Sd + 1
        out_h = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        out_w = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        num_patches = out_d * out_h * out_w
        patch_size = C * Kd * Kh * Kw

        # Ensure contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None

        # Allocate im2col matrix
        im2col = torch.empty((B * num_patches, patch_size), dtype=x.dtype, device=x.device)

        # Launch im2col kernel
        BLOCK_SIZE = 256
        grid_im2col = (B * num_patches,)
        im2col_kernel[grid_im2col](
            x, im2col,
            B, C, D, H, W,
            Kd, Kh, Kw,
            Sd, Sh, Sw,
            Pd, Ph, Pw,
            Dd, Dh, Dw,
            out_d, out_h, out_w,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # GEMM parameters
        M = B * num_patches
        N = self.out_channels
        K = patch_size

        # Transpose weight for GEMM: (K, N)
        weight_T = weight.view(N, K).t().contiguous()

        # Allocate output
        out_flat = torch.empty((M, N), dtype=x.dtype, device=x.device)

        # GEMM kernel parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        gemm_bias_kernel[grid_gemm](
            im2col, weight_T, bias, out_flat,
            M, N, K,
            im2col.stride(0), im2col.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )

        # Reshape output
        out = out_flat.view(B, self.out_channels, out_d, out_h, out_w)
        return out


def get_inputs():
    batch_size = 8
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    depth = 16
    height = 128
    width = 128
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]