import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_gemm_kernel(
    A_ptr,  # Pointer to input tensor A (flattened im2col)
    B_ptr,  # Pointer to weight tensor B (flattened weights)
    C_ptr,  # Pointer to output tensor C (flattened output)
    M,      # Number of rows in A (N * D_out * H_out * W_out)
    K,      # Number of columns in A (C_in * K^3)
    N,      # Number of columns in B (C_out)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M
    col_idx = tl.arange(0, BLOCK_N)
    col_mask = col_idx < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, 1):
        a_ptrs = A_ptr + row_idx * K + k
        a = tl.load(a_ptrs, mask=row_mask, other=0.0)
        b_ptrs = B_ptr + k * N + col_idx
        b = tl.load(b_ptrs, mask=col_mask, other=0.0)
        acc += tl.dot(tl.trans(a), b)

    c_ptrs = C_ptr + row_idx * N + col_idx
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Extract parameters
            in_channels = self.conv3d.in_channels
            out_channels = self.conv3d.out_channels
            kernel_size = self.conv3d.kernel_size[0]
            stride = self.conv3d.stride[0]
            padding = self.conv3d.padding[0]
            dilation = self.conv3d.dilation[0]
            groups = self.conv3d.groups
            weight = self.conv3d.weight  # Shape: (C_out, C_in, K, K, K)
            bias = self.conv3d.bias if self.conv3d.bias is not None else None

            # Compute output spatial dimensions
            depth, width, height = x.shape[2], x.shape[3], x.shape[4]
            depth_out = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

            # Reshape weight to (C_in * K^3, C_out)
            # w: (C_out, C_in, K, K, K) -> (C_out, C_in, K^3) -> (C_in, C_out, K^3) -> (C_in * K^3, C_out)
            weight_flat = weight.permute(1, 0, 2, 3, 4).reshape(in_channels * kernel_size**3, out_channels).contiguous()

            # Prepare output tensor
            out = torch.empty(x.shape[0], out_channels, depth_out, width_out, height_out, device=x.device, dtype=x.dtype)

            # Flatten output to (N * D_out * H_out * W_out, C_out)
            out_flat = out.permute(0, 2, 3, 4, 1).reshape(-1, out_channels).contiguous()

            # Number of rows in GEMM
            M = x.shape[0] * depth_out * width_out * height_out
            K_mat = in_channels * kernel_size**3
            N_mat = out_channels

            # Kernel parameters
            BLOCK_M = 16
            BLOCK_N = 64

            # Grid calculation
            grid = (M + BLOCK_M - 1) // BLOCK_M

            # Launch kernel
            conv3d_gemm_kernel[grid](
                x, weight_flat, out_flat,
                M, K_mat, N_mat,
                BLOCK_M, BLOCK_N
            )

            # Add bias if present
            if bias is not None:
                out_flat += bias

            return out


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]