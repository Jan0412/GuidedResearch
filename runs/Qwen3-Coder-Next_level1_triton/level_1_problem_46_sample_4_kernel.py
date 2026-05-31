import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    # Input tensor pointer
    x_ptr,
    # Output tensor pointer
    out_ptr,
    # Dimensions
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    D_in: tl.constexpr,  # input depth
    H_in: tl.constexpr,  # input height
    W_in: tl.constexpr,  # input width
    D_out: tl.constexpr,  # output depth
    H_out: tl.constexpr,  # output height
    W_out: tl.constexpr,  # output width
    # Pooling parameters
    K: tl.constexpr,  # kernel size
    S: tl.constexpr,  # stride
    P: tl.constexpr,  # padding
    # Block sizes for tiling
    BLOCK_C: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 4,
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_d = tl.program_id(2)  # depth output index
    pid_h = tl.program_id(3)  # height output index
    pid_w = tl.program_id(4)  # width output index

    # Calculate input indices for the pooling region
    d_start = pid_d * S - P
    h_start = pid_h * S - P
    w_start = pid_w * S - P

    # Compute valid range of pooling indices
    d_min = tl.maximum(d_start, 0)
    h_min = tl.maximum(h_start, 0)
    w_min = tl.maximum(w_start, 0)
    d_max = tl.minimum(d_start + K, D_in)
    h_max = tl.minimum(h_start + K, H_in)
    w_max = tl.minimum(w_start + K, W_in)

    # Calculate effective pooling size (number of valid elements)
    d_len = tl.maximum(0, d_max - d_min)
    h_len = tl.maximum(0, h_max - h_min)
    w_len = tl.maximum(0, w_max - w_min)
    pool_size = d_len * h_len * w_len

    # Offset to the start of this (n, c) in the input
    x_offset = pid_n * (C * D_in * H_in * W_in) + pid_c * (D_in * H_in * W_in)

    # Accumulator for sum
    sum_val = 0.0

    # Iterate through the pooling region
    for d in range(d_min, d_max):
        for h in range(h_min, h_max):
            for w in range(w_min, w_max):
                # Calculate input index
                idx = x_offset + d * (H_in * W_in) + h * W_in + w
                sum_val += tl.load(x_ptr + idx)

    # Compute average and store
    if pool_size > 0:
        avg = sum_val / pool_size
    else:
        avg = 0.0

    # Output index calculation
    out_offset = pid_n * (C * D_out * H_out * W_out) + pid_c * (D_out * H_out * W_out)
    out_idx = out_offset + pid_d * (H_out * W_out) + pid_h * W_out + pid_w

    tl.store(out_ptr + out_idx, avg)


def triton_avg_pool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
) -> torch.Tensor:
    """
    Applies 3D average pooling using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (N, C, D, H, W)
        kernel_size (int): Size of the kernel
        stride (int, optional): Stride of the pooling. Defaults to kernel_size.
        padding (int, optional): Padding. Defaults to 0.
        
    Returns:
        torch.Tensor: Output tensor after pooling.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    N, C, D_in, H_in, W_in = x.shape
    if stride is None:
        stride = kernel_size

    # Calculate output dimensions
    D_out = (D_in + 2 * padding - kernel_size) // stride + 1
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1

    # Allocate output tensor
    out = torch.empty(N, C, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # Define grid dimensions
    grid = (
        N,
        C,
        D_out,
        H_out,
        W_out,
    )

    # Launch kernel
    avg_pool3d_kernel[grid](
        x,
        out,
        N=N,
        C=C,
        D_in=D_in,
        H_in=H_in,
        W_in=W_in,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
        K=kernel_size,
        S=stride,
        P=padding,
        BLOCK_C=16,
        BLOCK_K=4,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom Triton kernel.
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
        Applies Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)