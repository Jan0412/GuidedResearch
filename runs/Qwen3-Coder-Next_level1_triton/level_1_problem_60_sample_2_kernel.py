import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for im2col transformation for 3D convolution
@triton.jit
def im2col_3d_kernel(
    x_ptr,  # Input tensor: (N, C, D, H, W)
    output_ptr,  # Output tensor: (N * D_out * H_out * W_out, C * kD * kH * kW)
    N, C, D, H, W,  # Input dimensions
    kD, kH, kW,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Strides
    pad_d, pad_h, pad_w,  # Padding
    dil_d, dil_h, dil_w,  # Dilation
    D_out, H_out, W_out,  # Output dimensions
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute indices
    n_idx = tl.program_id(0)
    k_idx = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Compute which sample in the batch
    sample_idx = n_idx // (D_out * H_out * W_out)
    pos_d = (n_idx // (H_out * W_out)) % D_out
    pos_h = (n_idx // W_out) % H_out
    pos_w = n_idx % W_out
    
    # Compute starting position in input
    start_d = pos_d * stride_d - pad_d
    start_h = pos_h * stride_h - pad_h
    start_w = pos_w * stride_w - pad_w
    
    # Compute output index in im2col matrix
    out_idx = n_idx * (C * kD * kH * kW) + k_idx
    
    # Compute input indices
    c_idx = k_idx // (kD * kH * kW)
    k_rest = k_idx % (kD * kH * kW)
    k_d = k_rest // (kH * kW)
    k_h = (k_rest // kW) % kH
    k_w = k_rest % kW
    
    # Compute input coordinates
    in_d = start_d + k_d * dil_d
    in_h = start_h + k_h * dil_h
    in_w = start_w + k_w * dil_w
    
    # Create masks
    c_mask = c_idx < C
    valid_mask = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
    full_mask = c_mask & (k_idx < C * kD * kH * kW) & valid_mask
    
    # Compute input pointer offset
    offset = sample_idx * (C * D * H * W) + c_idx * (D * H * W) + in_d * (H * W) + in_h * W + in_w
    
    # Load data and store to im2col output
    data = tl.load(x_ptr + offset, mask=full_mask, other=0.0)
    tl.store(output_ptr + out_idx, data, mask=full_mask)


# Triton kernel for GEMM (matrix multiplication)
@triton.jit
def gemm_kernel(
    A_ptr,  # Input matrix A: (M, K)
    B_ptr,  # Input matrix B: (K, N)
    C_ptr,  # Output matrix C: (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped program IDs for better cache performance
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    group_id = pid // num_programs_in_group
    first_program_m = group_id * GROUP_SIZE_M
    program_ids_m = tl.arange(0, GROUP_SIZE_M)
    program_ids_m = tl.maximum(first_program_m, tl.minimum(program_ids_m, num_programs_m - 1))
    program_ids_n = pid % num_programs_n
    
    # Create block offsets
    offs_m = program_ids_m[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = program_ids_n[None, :] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A block
        a_offset = offs_m[:, None] * stride_am + (offs_k[None, :] + k * BLOCK_SIZE_K) * stride_ak
        a = tl.load(A_ptr + a_offset, mask=a_mask, other=0.0)
        
        # Load B block
        b_offset = (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(B_ptr + b_offset, mask=b_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to output type and store
    c = accumulator.to(tl.float32)
    
    c_offset = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + c_offset, c, mask=c_mask)


# Triton kernel for adding bias
@triton.jit
def add_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load data
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Add bias
    out = x + bias
    
    # Store result
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """
    Performs 3D convolution using Triton kernels with im2col approach.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C, D, H, W = x.shape
    out_channels, in_channels_per_group, kD, kH, kW = weight.shape
    
    # Compute output dimensions
    stride_d = stride_h = stride_w = stride if isinstance(stride, int) else stride
    pad_d = pad_h = pad_w = padding if isinstance(padding, int) else padding
    dil_d = dil_h = dil_w = dilation if isinstance(dilation, int) else dilation
    
    D_out = (D + 2 * pad_d - dil_d * (kD - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    
    # Compute im2col output size
    K = C * kD * kH * kW
    M = N * D_out * H_out * W_out
    N_out = out_channels
    
    # Create im2col output tensor
    im2col_output = torch.empty((M, K), device=x.device, dtype=x.dtype)
    
    # Launch im2col kernel
    BLOCK_SIZE_N = 128  # Number of output positions per block
    BLOCK_SIZE_K = 128  # Number of kernel elements per block
    
    grid_im2col = (
        (M + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )
    
    im2col_3d_kernel[grid_im2col](
        x, im2col_output,
        N, C, D, H, W,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        D_out, H_out, W_out,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=1,  # Not used in this kernel
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Reshape weight for GEMM: (out_channels, K)
    weight_reshaped = weight.view(out_channels, K).t().contiguous()  # (K, out_channels)
    
    # Create output tensor for GEMM
    gemm_output = torch.empty((M, N_out), device=x.device, dtype=x.dtype)
    
    # Launch GEMM kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    grid_gemm = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N_out, BLOCK_SIZE_N),
    )
    
    gemm_kernel[grid_gemm](
        im2col_output, weight_reshaped, gemm_output,
        M, N_out, K,
        im2col_output.stride(0), im2col_output.stride(1),
        weight_reshaped.stride(0), weight_reshaped.stride(1),
        gemm_output.stride(0), gemm_output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    # Add bias if present
    if bias is not None:
        bias_output = torch.empty_like(gemm_output)
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        
        grid_bias = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N_out, BLOCK_SIZE_N)
        )
        
        add_bias_kernel[grid_bias](
            gemm_output, bias, bias_output,
            M, N_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        gemm_output = bias_output
    
    # Reshape output to (N, out_channels, D_out, H_out, W_out)
    output = gemm_output.view(N, D_out, H_out, W_out, out_channels)
    output = output.permute(0, 4, 1, 2, 3).contiguous()  # (N, out_channels, D_out, H_out, W_out)
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of the 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernels.
        """
        # Extract parameters from the original conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        
        stride = self.conv3d.stride[0] if isinstance(self.conv3d.stride, tuple) else self.conv3d.stride
        padding = self.conv3d.padding[0] if isinstance(self.conv3d.padding, tuple) else self.conv3d.padding
        dilation = self.conv3d.dilation[0] if isinstance(self.conv3d.dilation, tuple) else self.conv3d.dilation
        groups = self.conv3d.groups
        
        # Use our Triton implementation
        return triton_conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)