import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool3d_kernel(
    x_ptr,  # Input tensor pointer: (B, C, D, H, W)
    out_ptr,  # Output tensor pointer: (B, C, D_out, H_out, W_out)
    n_batches, n_channels, n_depth, n_height, n_width,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID (for 1D grid)
    pid = tl.program_id(0)
    
    # Total number of output elements
    total_out = n_batches * n_channels * out_d * out_h * out_w
    
    # Compute indices for each dimension
    # Layout: [batch, channel, depth_out, height_out, width_out]
    # We'll flatten everything to 1D and then decode indices
    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < total_out
    
    # Decode flattened index into (b, c, d_out, h_out, w_out)
    w_out_idx = out_idx % out_w
    h_out_idx = (out_idx // out_w) % out_h
    d_out_idx = (out_idx // (out_w * out_h)) % out_d
    c_idx = (out_idx // (out_w * out_h * out_d)) % n_channels
    b_idx = out_idx // (out_w * out_h * out_d * n_channels)
    
    # Compute input start positions for the pooling window
    d_in_start = d_out_idx * stride_d - pad_d
    h_in_start = h_out_idx * stride_h - pad_h
    w_in_start = w_out_idx * stride_w - pad_w
    
    # Accumulator for average
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over pooling kernel window
    for kd in range(kernel_d):
        d_in = d_in_start + kd
        d_in_valid = (d_in >= 0) & (d_in < n_depth)
        for kh in range(kernel_h):
            h_in = h_in_start + kh
            h_in_valid = (h_in >= 0) & (h_in < n_height)
            for kw in range(kernel_w):
                w_in = w_in_start + kw
                w_in_valid = (w_in >= 0) & (w_in < n_width)
                
                # Compute input index
                in_idx = (
                    b_idx * (n_channels * n_depth * n_height * n_width) +
                    c_idx * (n_depth * n_height * n_width) +
                    d_in * (n_height * n_width) +
                    h_in * n_width +
                    w_in
                )
                
                # Load input value with mask for valid indices only
                in_val = tl.load(
                    x_ptr + in_idx,
                    mask=(d_in_valid & h_in_valid & w_in_valid),
                    other=0.0
                )
                
                # Accumulate
                acc += in_val
                count += (d_in_valid & h_in_valid & w_in_valid).to(tl.float32)
    
    # Compute average (handle zero count case to avoid division by zero)
    avg = tl.where(count > 0, acc / count, 0.0)
    
    # Store result
    tl.store(out_ptr + out_idx, avg.to(x_ptr.dtype.element_ty), mask=mask)


def triton_avgpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0
) -> torch.Tensor:
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
        kernel_size: Size of the pooling kernel
        stride: Stride for pooling (defaults to kernel_size)
        padding: Padding to apply before pooling
    
    Returns:
        Output tensor after pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, depth, height, width = x.shape
    
    # Determine effective stride and kernel size
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    out_d = (depth + 2 * padding - kernel_size) // stride + 1
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty(
        batch_size, channels, out_d, out_h, out_w,
        dtype=x.dtype, device=x.device
    )
    
    # Handle edge case where output is empty
    if out_d <= 0 or out_h <= 0 or out_w <= 0:
        return torch.zeros_like(out)
    
    # Kernel parameters
    kernel_d = kernel_h = kernel_w = kernel_size
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding
    
    # Compute total output elements
    total_out = batch_size * channels * out_d * out_h * out_w
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid size
    grid = lambda meta: (
        triton.cdiv(total_out, meta["BLOCK_SIZE"]),
    )
    
    # Launch kernel
    avgpool3d_kernel[grid](
        x, out,
        batch_size, channels, depth, height, width,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding
        )