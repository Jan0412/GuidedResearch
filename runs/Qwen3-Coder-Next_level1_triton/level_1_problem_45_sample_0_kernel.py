import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    in_h,  # Input height
    in_w,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    BLOCK_SIZE: tl.constexpr,
    # Tile sizes for blocking to improve cache usage
    TILE_C: tl.constexpr = 1,  # Process one channel at a time (simpler, avoids race conditions)
    TILE_H: tl.constexpr = 16,
    TILE_W: tl.constexpr = 16,
):
    # Compute the batch index and channel index
    bc_id = tl.program_id(0)
    b = bc_id // channels
    c = bc_id % channels

    # Compute the output spatial position
    out_h_id = tl.program_id(1)
    out_w_id = tl.program_id(2)

    # Compute the input region this thread block is responsible for
    h_start = out_h_id * stride_h - pad_h
    w_start = out_w_id * stride_w - pad_w

    # Compute the effective region (clamped to input bounds)
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end_clamped = tl.minimum(h_start + kernel_h, in_h)
    w_end_clamped = tl.minimum(w_start + kernel_w, in_w)

    # Only process if the region is valid
    if h_end_clamped <= h_start_clamped or w_end_clamped <= w_start_clamped:
        # No valid region: output is zero (padding or out-of-bounds)
        out_idx = (
            b * channels * out_h * out_w +
            c * out_h * out_w +
            out_h_id * out_w +
            out_w_id
        )
        tl.store(out_ptr + out_idx, 0.0)
        return

    # Compute actual kernel dimensions
    h_start_eff = tl.maximum(0, -h_start)
    w_start_eff = tl.maximum(0, -w_start)
    actual_h = h_end_clamped - h_start_clamped
    actual_w = w_end_clamped - w_start_clamped

    # Accumulator for sum
    sum_val = 0.0
    count = 0

    # Iterate over the pooling region
    for h_offset in range(h_start_eff, actual_h):
        in_h_idx = h_start_clamped + h_offset
        for w_offset in range(w_start_eff, actual_w):
            in_w_idx = w_start_clamped + w_offset

            # Compute input pointer offset
            in_idx = (
                b * channels * in_h * in_w +
                c * in_h * in_w +
                in_h_idx * in_w +
                in_w_idx
            )
            val = tl.load(x_ptr + in_idx)
            sum_val += val
            count += 1

    # Compute average
    avg = sum_val / count

    # Store result
    out_idx = (
        b * channels * out_h * out_w +
        c * out_h * out_w +
        out_h_id * out_w +
        out_w_id
    )
    tl.store(out_ptr + out_idx, avg)


def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
):
    """
    Triton implementation of 2D Average Pooling.
    """
    # Ensure input is contiguous and on CUDA
    x = x.contiguous()
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    # Extract dimensions
    batch_size, channels, in_h, in_w = x.shape

    # Compute output dimensions
    stride = stride if stride is not None else kernel_size
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Grid: [batch_size * channels, out_h, out_w]
    grid = (batch_size * channels, out_h, out_w)

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
        BLOCK_SIZE=128,
        TILE_C=1,
        TILE_H=16,
        TILE_W=16,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )