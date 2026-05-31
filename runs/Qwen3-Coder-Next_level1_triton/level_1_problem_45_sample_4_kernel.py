import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    N,  # Batch size
    C,  # Number of channels
    H,  # Input height
    W,  # Input width
    out_H,  # Output height
    out_W,  # Output width
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output batch, channel, height, width indices
    out_idx = tl.program_id(0)
    
    # Decompose out_idx into (n, c, h_out, w_out)
    # Total number of output elements per (h_out, w_out) for one (n, c)
    total_out_hw = out_H * out_W
    n_c = out_idx // total_out_hw
    hw = out_idx % total_out_hw
    n = n_c // C
    c = n_c % C
    h_out = hw // out_W
    w_out = hw % out_W
    
    # Compute the top-left corner of the pooling window in input
    h_start = h_out * stride - padding
    w_start = w_out * stride - padding
    
    # Compute the actual pooling region boundaries (clamped to input bounds)
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end = h_start + kernel_size
    w_end = w_start + kernel_size
    h_end_clamped = tl.minimum(h_end, H)
    w_end_clamped = tl.minimum(w_end, W)
    
    # Calculate how many valid elements are in the pooling region
    valid_height = h_end_clamped - h_start_clamped
    valid_width = w_end_clamped - w_start_clamped
    valid_count = valid_height * valid_width
    
    # Compute the starting pointer for this (n, c) in the input
    # Input shape: (N, C, H, W), row-major layout
    input_offset = (n * C * H * W) + (c * H * W)
    
    # Accumulate sum over the pooling window
    sum_val = 0.0
    for h in range(h_start_clamped, h_end_clamped):
        for w in range(w_start_clamped, w_end_clamped):
            # Compute input index
            input_idx = h * W + w
            # Load the value
            x_val = tl.load(x_ptr + input_offset + input_idx)
            sum_val += x_val
    
    # Compute average
    avg_val = sum_val / valid_count
    
    # Store result
    out_offset = out_idx  # Since output is (N, C, out_H, out_W) and we're using flattened index
    tl.store(out_ptr + out_offset, avg_val)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Applies 2D Average Pooling using a Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation (defaults to kernel_size)
        padding: Padding applied to the input
    
    Returns:
        Output tensor after average pooling
    """
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Parse parameters
    N, C, H, W = x.shape
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)
    
    # Flatten the output for easier indexing (N * C * out_H * out_W elements)
    out_flat_size = N * C * out_H * out_W
    
    # Define block size (tunable)
    BLOCK_SIZE = 128
    
    # Grid: one block per output element (we'll handle this via flattened indexing)
    grid = (out_flat_size,)
    
    # Launch the kernel
    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        out_H, out_W,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton kernels.
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
        Applies 2D Average Pooling to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)