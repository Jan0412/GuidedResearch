import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr,
    out_ptr,
    stride_B, stride_C, stride_D, stride_H, stride_W,
    out_stride_B, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
    B, C, D, H, W,
    D_out, H_out, W_out,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    # Map program ID to output coordinates
    # Grid: (B, C, D_out, H_out, (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    b = tl.program_id(0)
    c = tl.program_id(1)
    d = tl.program_id(2)
    h = tl.program_id(3)
    w_start = tl.program_id(4) * BLOCK_SIZE
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE)
    mask_w = w_offsets < W_out

    # Base pointer for the current batch and channel
    # x_ptr + b * stride_B + c * stride_C
    base_ptr = x_ptr + b * stride_B + c * stride_C

    # Accumulator for the average pooling
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the kernel window
    for kd in range(KERNEL_SIZE):
        id_val = d * stride + kd - padding
        if 0 <= id_val < D:
            for kh in range(KERNEL_SIZE):
                ih_val = h * stride + kh - padding
                if 0 <= ih_val < H:
                    for kw in range(KERNEL_SIZE):
                        # Calculate input width indices for the block of output widths
                        iw_val = w_offsets * stride + kw - padding
                        # Mask for width boundaries and the output block boundary
                        mask_iw = (iw_val >= 0) & (iw_val < W) & mask_w
                        
                        # Calculate the linear offset for the input tensor
                        offset = id_val * stride_D + ih_val * stride_H + iw_val * stride_W
                        val = tl.load(base_ptr + offset, mask=mask_iw, other=0.0)
                        acc += val

    # Average pooling: divide by kernel_size^3 (count_include_pad=True)
    acc /= (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)

    # Calculate output pointer and store results
    out_base_ptr = out_ptr + b * out_stride_B + c * out_stride_C + d * out_stride_D + h * out_stride_H
    tl.store(out_base_ptr + w_offsets * out_stride_W, acc, mask=mask_w)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape

    # Calculate output dimensions
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Input strides
    stride_B = C * D * H * W
    stride_C = D * H * W
    stride_D = H * W
    stride_H = W
    stride_W = 1

    # Output strides
    out_stride_B = C * D_out * H_out * W_out
    out_stride_C = D_out * H_out * W_out
    out_stride_D = H_out * W_out
    out_stride_H = W_out
    out_stride_W = 1

    BLOCK_SIZE = 16
    grid = (B, C, D_out, H_out, (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)

    avg_pool3d_kernel[grid](
        x, out,
        stride_B, stride_C, stride_D, stride_H, stride_W,
        out_stride_B, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
        B, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)