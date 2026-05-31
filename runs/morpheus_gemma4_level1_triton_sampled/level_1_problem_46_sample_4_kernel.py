import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr,
    N, C, D, H, W,
    D_out, H_out, W_out,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid is (N, C, D_out, H_out, (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    n = tl.program_id(0)
    c = tl.program_id(1)
    d = tl.program_id(2)
    h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    w_offsets = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = w_offsets < W_out
    
    # Accumulator for the sum of the pooling window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over the 3D kernel window
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                # Calculate input coordinates
                in_d = d * stride + i - padding
                in_h = h * stride + j - padding
                in_w = w_offsets * stride + k - padding
                
                # Boundary check for depth and height (scalar)
                if in_d >= 0 and in_d < D and in_h >= 0 and in_h < H:
                    # Boundary check for width (vectorized)
                    w_mask = (in_w >= 0) & (in_w < W) & mask
                    
                    # Calculate pointer to the input element
                    # x_ptr is the base, then offset by (n, c, in_d, in_h, in_w)
                    ptr = (x_ptr + 
                           n * stride_n + 
                           c * stride_c + 
                           in_d * stride_d + 
                           in_h * stride_h + 
                           in_w * stride_w)
                    
                    # Load and accumulate
                    sum_val += tl.load(ptr, mask=w_mask, other=0.0)
    
    # Calculate output pointer base for the current (n, c, d, h)
    out_ptr_base = (out_ptr + 
                    n * out_stride_n + 
                    c * out_stride_c + 
                    d * out_stride_d + 
                    h * out_stride_h)
    
    # Average the sum and store the result
    # PyTorch's AvgPool3d default is count_include_pad=True, so we divide by kernel_size^3
    divisor = float(kernel_size**3)
    tl.store(out_ptr_base + w_offsets * out_stride_w, sum_val / divisor, mask=mask)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Get strides for indexing
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    BLOCK_SIZE = 128
    # Grid: (N, C, D_out, H_out, ceil(W_out / BLOCK_SIZE))
    grid = (N, C, D_out, H_out, (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    avg_pool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        D_out, H_out, W_out,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)