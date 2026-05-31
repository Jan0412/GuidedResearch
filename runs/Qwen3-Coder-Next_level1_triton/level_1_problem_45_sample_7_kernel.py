import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    n_batches,
    n_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Compute output spatial indices
    out_batch = tl.program_id(0)
    out_ch = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Compute input spatial indices (top-left corner of pooling window)
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Compute valid pooling region
    pool_h_start = tl.maximum(0, -in_h_start)
    pool_h_end = tl.minimum(kernel_h, in_h - in_h_start)
    pool_w_start = tl.maximum(0, -in_w_start)
    pool_w_end = tl.minimum(kernel_w, in_w - in_w_start)
    
    # Accumulator for sum
    acc = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over pooling window
    for kh in range(pool_h_start, pool_h_end):
        h_idx = in_h_start + kh
        for kw in range(pool_w_start, pool_w_end):
            w_idx = in_w_start + kw
            # Compute input pointer offset
            x_offset = (
                out_batch * n_channels * in_h * in_w +
                out_ch * in_h * in_w +
                h_idx * in_w +
                w_idx
            )
            x_val = tl.load(x_ptr + x_offset)
            acc += x_val
            count += 1
    
    # Compute average and store result
    if count > 0:
        avg = acc / tl.cast(count, tl.float32)
    else:
        avg = tl.zeros((1,), dtype=tl.float32)
    
    out_offset = (
        out_batch * n_channels * out_h * out_w +
        out_ch * out_h * out_w +
        out_h_idx * out_w +
        out_w_idx
    )
    tl.store(out_ptr + out_offset, avg)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Triton implementation of 2D Average Pooling.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        kernel_size (int): Size of the pooling window
        stride (int, optional): Stride of the pooling operation (default: kernel_size)
        padding (int, optional): Padding applied to the input tensor (default: 0)
        
    Returns:
        torch.Tensor: Output tensor after applying average pooling
    """
    batch_size, channels, in_h, in_w = x.shape
    
    if stride is None:
        stride = kernel_size
        
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Set block sizes for efficient parallelization
    # We'll parallelize over batch, channel, and spatial dimensions
    BLOCK_H = min(32, out_h)
    BLOCK_W = min(32, out_w)
    
    # Grid dimensions: [batch, channels, out_h//BLOCK_H, out_w//BLOCK_W]
    grid = (
        batch_size,
        channels,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x,
        out,
        batch_size,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size,
        kernel_size,
        stride,
        stride,
        padding,
        padding,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)