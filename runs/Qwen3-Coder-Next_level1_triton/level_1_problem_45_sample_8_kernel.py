import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool2d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    N, C, H, W,  # Input dimensions
    out_h, out_w,  # Output spatial dimensions
    pool_h, pool_w,  # Pooling kernel height and width
    stride_h, stride_w,  # Stride in height and width
    pad_h, pad_w,  # Padding in height and width
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output batch index
    batch_idx = tl.program_id(0)
    # Compute output channel index
    channel_idx = tl.program_id(1)
    # Compute output spatial position (flattened)
    out_pos = tl.program_id(2)
    
    # Convert flattened position to (oh, ow)
    oh = out_pos // out_w
    ow = out_pos % out_w
    
    # Compute input coordinates for the top-left corner of the pooling window
    h_start = oh * stride_h - pad_h
    w_start = ow * stride_w - pad_w
    
    # Compute actual pooling window boundaries (clamped to input bounds)
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end_clamped = tl.minimum(h_start + pool_h, H)
    w_end_clamped = tl.minimum(w_start + pool_w, W)
    
    # Compute the actual pooling window size (might be smaller at boundaries)
    actual_pool_h = h_end_clamped - h_start_clamped
    actual_pool_w = w_end_clamped - w_start_clamped
    pool_area = actual_pool_h * actual_pool_w
    
    # Compute input pointer offset for this batch and channel
    input_offset = batch_idx * C * H * W + channel_idx * H * W
    
    # Accumulator for sum
    sum_val = 0.0
    
    # Iterate over the pooling window
    for ph in range(actual_pool_h):
        h_idx = h_start_clamped + ph
        for pw in range(actual_pool_w):
            w_idx = w_start_clamped + pw
            # Compute input index
            input_idx = input_offset + h_idx * W + w_idx
            sum_val += tl.load(x_ptr + input_idx)
    
    # Compute average
    avg_val = sum_val / pool_area
    
    # Compute output pointer offset
    output_offset = batch_idx * C * out_h * out_w + channel_idx * out_h * out_w
    output_idx = output_offset + oh * out_w + ow
    
    # Store result
    tl.store(y_ptr + output_idx, avg_val)


def triton_avgpool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0) -> torch.Tensor:
    """
    Applies 2D Average Pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation (default: kernel_size)
        padding: Padding applied to the input tensor (default: 0)
    
    Returns:
        Output tensor with Average Pooling applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (batch_size, channels, out_height * out_width)
    
    # Launch kernel
    avgpool2d_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        out_height, out_width,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        BLOCK_SIZE=128,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton kernel.
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
        return triton_avgpool2d(x, self.kernel_size, self.stride, self.padding)